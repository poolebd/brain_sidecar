from __future__ import annotations

import importlib
import os
import subprocess
import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str
    used_memory_mb: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_name": self.process_name,
            "used_memory_mb": self.used_memory_mb,
        }


@dataclass(frozen=True)
class GpuStatus:
    nvidia_available: bool
    name: str | None
    memory_total_mb: int | None
    memory_used_mb: int | None
    memory_free_mb: int | None
    driver_version: str | None
    gpu_pressure: str
    gpu_processes: list[GpuProcess]
    asr_cuda_available: bool
    asr_cuda_error: str | None
    ollama_gpu_models: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nvidia_available": self.nvidia_available,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "driver_version": self.driver_version,
            "gpu_pressure": self.gpu_pressure,
            "gpu_processes": [process.to_dict() for process in self.gpu_processes],
            "asr_cuda_available": self.asr_cuda_available,
            "asr_cuda_error": self.asr_cuda_error,
            "ollama_gpu_models": self.ollama_gpu_models,
        }


@dataclass(frozen=True)
class AsrGpuPreparation:
    before: GpuStatus
    after: GpuStatus
    attempted_unload: bool
    stopped_ollama_models: list[str]
    errors: list[str]


def _run(
    args: list[str],
    *,
    timeout: float = 5,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout, env=env)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _nvidia_status() -> tuple[bool, str | None, int | None, int | None, int | None, str | None]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return False, None, None, None, None, None
    first_line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 5:
        return True, first_line, None, None, None, None
    name, total, used, free, driver = parts
    return True, name, _parse_mb(total), _parse_mb(used), _parse_mb(free), driver


def _parse_mb(value: str) -> int | None:
    try:
        return int(float(value.strip()))
    except ValueError:
        return None


def _asr_cuda_status() -> tuple[bool, str | None]:
    _bootstrap_python_cuda_lib_path()
    try:
        import ctranslate2
    except Exception as exc:
        return False, f"ctranslate2 import failed: {exc}"

    try:
        count_getter = getattr(ctranslate2, "get_cuda_device_count", None)
        if count_getter is not None and count_getter() > 0:
            return True, None
        return False, "ctranslate2 reported zero CUDA devices"
    except Exception as exc:
        return False, f"ctranslate2 CUDA check failed: {exc}"


def _gpu_processes() -> list[GpuProcess]:
    result = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None or result.returncode != 0:
        return []
    processes: list[GpuProcess] = []
    for line in result.stdout.splitlines():
        process = _parse_gpu_process(line)
        if process is not None:
            processes.append(process)
    return processes


def _parse_gpu_process(line: str) -> GpuProcess | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    pid, process_name, used_memory = parts
    try:
        return GpuProcess(
            pid=int(pid),
            process_name=process_name,
            used_memory_mb=int(float(used_memory)),
        )
    except ValueError:
        return None


def gpu_pressure(memory_total_mb: int | None, memory_free_mb: int | None) -> str:
    if memory_total_mb is None or memory_free_mb is None:
        return "unknown"
    if memory_free_mb < 1600:
        return "critical"
    if memory_free_mb < 3500 or memory_free_mb / max(memory_total_mb, 1) < 0.25:
        return "tight"
    return "ok"


def parse_ollama_gpu_models(output: str) -> list[str]:
    models: list[str] = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if parts and "GPU" in line:
            models.append(parts[0])
    return list(dict.fromkeys(models))


def _ollama_gpu_models() -> list[str]:
    result = _run(["ollama", "ps"])
    if result is None or result.returncode != 0:
        return []
    return parse_ollama_gpu_models(result.stdout)


def read_gpu_status() -> GpuStatus:
    nvidia_available, name, total, used, free, driver = _nvidia_status()
    asr_ok, asr_error = _asr_cuda_status()
    return GpuStatus(
        nvidia_available=nvidia_available,
        name=name,
        memory_total_mb=total,
        memory_used_mb=used,
        memory_free_mb=free,
        driver_version=driver,
        gpu_pressure=gpu_pressure(total, free),
        gpu_processes=_gpu_processes() if nvidia_available else [],
        asr_cuda_available=asr_ok,
        asr_cuda_error=asr_error,
        ollama_gpu_models=_ollama_gpu_models(),
    )


def require_asr_cuda() -> None:
    status = read_gpu_status()
    if not status.nvidia_available:
        raise RuntimeError("NVIDIA GPU is not visible to nvidia-smi.")
    if not status.asr_cuda_available:
        raise RuntimeError(status.asr_cuda_error or "ASR CUDA runtime is unavailable.")


def prepare_asr_gpu(settings: Any, *, force_unload: bool = False) -> AsrGpuPreparation:
    before = read_gpu_status()
    if not before.nvidia_available:
        raise RuntimeError("NVIDIA GPU is not visible to nvidia-smi.")
    if not before.asr_cuda_available:
        raise RuntimeError(before.asr_cuda_error or "ASR CUDA runtime is unavailable.")

    min_free_mb = int(getattr(settings, "asr_min_free_vram_mb", 3500))
    unload_enabled = bool(getattr(settings, "asr_unload_ollama_on_start", True))
    timeout_s = float(getattr(settings, "asr_gpu_free_timeout_seconds", 10.0))
    ollama_host = str(getattr(settings, "ollama_host", ""))
    needs_free = not _has_min_free_vram(before, min_free_mb)
    attempted_unload = False
    stopped_models: list[str] = []
    errors: list[str] = []
    after = before

    if unload_enabled and (force_unload or needs_free) and before.ollama_gpu_models:
        attempted_unload = True
        stopped_models, errors = stop_ollama_gpu_models(ollama_host, before.ollama_gpu_models)
        after = _wait_for_free_vram(min_free_mb, timeout_s)

    if not _has_min_free_vram(after, min_free_mb):
        raise RuntimeError(_insufficient_vram_message(settings, before, after, attempted_unload, stopped_models, errors))

    return AsrGpuPreparation(
        before=before,
        after=after,
        attempted_unload=attempted_unload,
        stopped_ollama_models=stopped_models,
        errors=errors,
    )


def stop_ollama_gpu_models(ollama_host: str, models: list[str]) -> tuple[list[str], list[str]]:
    env = os.environ.copy()
    cli_host = _ollama_cli_host(ollama_host)
    if cli_host:
        env["OLLAMA_HOST"] = cli_host

    stopped: list[str] = []
    errors: list[str] = []
    for model in list(dict.fromkeys(models)):
        result = _run(["ollama", "stop", model], timeout=10, env=env)
        if result is None:
            errors.append(f"{model}: ollama stop timed out or ollama CLI was not found")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            errors.append(f"{model}: {detail}")
            continue
        stopped.append(model)
    return stopped, errors


def cuda_out_of_memory(exc: BaseException | str) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message


def _has_min_free_vram(status: GpuStatus, min_free_mb: int) -> bool:
    if min_free_mb <= 0 or status.memory_free_mb is None:
        return True
    return status.memory_free_mb >= min_free_mb


def _wait_for_free_vram(min_free_mb: int, timeout_s: float) -> GpuStatus:
    deadline = time.monotonic() + timeout_s
    latest = read_gpu_status()
    while not _has_min_free_vram(latest, min_free_mb) and time.monotonic() < deadline:
        time.sleep(0.3)
        latest = read_gpu_status()
    return latest


def _insufficient_vram_message(
    settings: Any,
    before: GpuStatus,
    after: GpuStatus,
    attempted_unload: bool,
    stopped_models: list[str],
    errors: list[str],
) -> str:
    min_free_mb = int(getattr(settings, "asr_min_free_vram_mb", 3500))
    free_mb = after.memory_free_mb if after.memory_free_mb is not None else "unknown"
    if attempted_unload:
        if stopped_models:
            action = f"Stopped GPU-resident Ollama models first: {', '.join(stopped_models)}."
        elif errors:
            action = "Tried to stop GPU-resident Ollama models, but the stop command failed."
        else:
            action = "Tried to stop GPU-resident Ollama models first."
    elif not bool(getattr(settings, "asr_unload_ollama_on_start", True)):
        action = "Automatic Ollama unload is disabled."
    elif before.ollama_gpu_models:
        action = f"Ollama GPU models were detected: {', '.join(before.ollama_gpu_models)}."
    else:
        action = "No GPU-resident Ollama model was detected to unload."

    process_summary = _process_summary(after.gpu_processes)
    error_summary = f" Ollama stop errors: {'; '.join(errors)}." if errors else ""
    return (
        f"Could not reserve enough free GPU memory for CUDA Faster-Whisper. "
        f"Need at least {min_free_mb} MB free VRAM before loading ASR; currently free: {free_mb} MB. "
        f"{action}{error_summary} GPU processes: {process_summary}."
    )


def _process_summary(processes: list[GpuProcess]) -> str:
    if not processes:
        return "none reported by nvidia-smi"
    top = sorted(processes, key=lambda process: process.used_memory_mb, reverse=True)[:5]
    return ", ".join(
        f"{Path(process.process_name).name} pid {process.pid} {process.used_memory_mb} MB" for process in top
    )


def _ollama_cli_host(ollama_host: str) -> str:
    if not ollama_host:
        return ""
    parsed = urlparse(ollama_host)
    if parsed.netloc:
        return parsed.netloc
    return ollama_host.removeprefix("http://").removeprefix("https://").rstrip("/")


def _bootstrap_python_cuda_lib_path() -> None:
    """Expose pip-installed NVIDIA runtime libraries before CTranslate2 loads."""

    paths: list[Path] = []
    for module_name in ("nvidia.cublas.lib", "nvidia.cuda_nvrtc.lib", "nvidia.cudnn.lib"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        module_file = getattr(module, "__file__", None)
        if module_file:
            paths.append(Path(module_file).parent)
        for module_path in getattr(module, "__path__", []):
            paths.append(Path(str(module_path)))

    if not paths:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_parts = [part for part in existing.split(":") if part]
    unique_paths: list[str] = []
    for path in paths:
        path_str = str(path)
        if path_str and path_str not in unique_paths:
            unique_paths.append(path_str)
    merged = [path for path in unique_paths if path not in existing_parts] + existing_parts
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    for library in (
        "libnvrtc.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
    ):
        for path in paths:
            candidate = path / library
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break
