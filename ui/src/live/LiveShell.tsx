import type { ReactNode } from "react";

export type LiveWorkbenchMode = "live" | "ended" | "replay" | "dev-test";

export type SessionRuntimeItem = {
  label: string;
  value: string;
  tone?: "normal" | "good" | "warning";
};

export function LiveMeetingWorkbench({
  mode,
  active,
  transcriptPane,
  sidecarPane,
}: {
  mode: LiveWorkbenchMode;
  active: boolean;
  transcriptPane: ReactNode;
  sidecarPane: ReactNode;
}) {
  return (
    <main
      className={`workspace-grid page live-page live-workbench mode-${mode} ${active ? "is-running" : ""}`}
      aria-label="Live"
      data-live-mode={mode}
    >
      {transcriptPane}
      {sidecarPane}
    </main>
  );
}

export function SessionRuntimeStrip({ mode, items }: { mode: LiveWorkbenchMode; items: SessionRuntimeItem[] }) {
  return (
    <section className="session-runtime-strip" aria-label="Session runtime strip">
      <span className={`runtime-mode-chip mode-${mode}`}>{liveWorkbenchModeLabel(mode)}</span>
      {items.map((item) => (
        <span key={`${item.label}-${item.value}`} className={`runtime-strip-item ${item.tone ?? "normal"}`}>
          <strong>{item.label}</strong>
          {item.value}
        </span>
      ))}
    </section>
  );
}

export function SessionSelectorBar({
  title,
  status,
  disabled,
  onTitleChange,
}: {
  title: string;
  status: string;
  disabled: boolean;
  onTitleChange: (title: string) => void;
}) {
  return (
    <section className="session-workspace-bar" aria-label="Live setup">
      <label className="field compact-field session-title-field">
        <span>Title</span>
        <input
          aria-label="Live session title"
          value={title}
          disabled={disabled}
          onChange={(event) => onTitleChange(event.target.value)}
          placeholder="New meeting"
        />
        <small>{status === "idle" || status === "stopped" ? "standby" : status} · Live output is ephemeral</small>
      </label>
      <div className="session-static-session" aria-label="Live run">
        <span>Live run</span>
        <strong>Ephemeral</strong>
      </div>
      <div className="session-static-session" aria-label="Live retention">
        <span>Review handoff</span>
        <strong>Audio temporary until validation</strong>
      </div>
    </section>
  );
}

export function liveWorkbenchModeLabel(mode: LiveWorkbenchMode): string {
  if (mode === "ended") return "Ended";
  if (mode === "replay") return "Replay";
  if (mode === "dev-test") return "Test";
  return "Live";
}
