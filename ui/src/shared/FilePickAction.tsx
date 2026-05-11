import type { ChangeEvent } from "react";

function pathLabel(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) ?? path;
}

export function FilePickAction({
  label,
  actionLabel,
  accept,
  path,
  busy,
  disabled,
  onChange,
}: {
  label: string;
  actionLabel: string;
  accept: string;
  path: string;
  busy: boolean;
  disabled: boolean;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <label className={`file-action ${disabled ? "disabled" : ""} ${busy ? "busy" : ""}`}>
      <input
        aria-label={label}
        className="file-picker-native"
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={onChange}
      />
      <span>{busy ? "Uploading..." : actionLabel}</span>
      <small title={path || "No local file selected"}>{path ? pathLabel(path) : "No local file selected"}</small>
    </label>
  );
}
