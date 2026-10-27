const CHIP =
  "inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-xs text-ink-700 shadow-chip";

interface StatusBarProps {
  connected: boolean;
  lastDevice: string;
  lastTimestamp: string;
}

export default function StatusBar({
  connected,
  lastDevice,
  lastTimestamp,
}: StatusBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className={CHIP}>
        Socket:{" "}
        <span
          className={
            connected ? "font-semibold text-safe" : "font-semibold text-risk"
          }
        >
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>
      <div className={CHIP}>
        Last Device: <span>{lastDevice}</span>
      </div>
      <div className={CHIP}>
        Timestamp: <span>{lastTimestamp}</span>
      </div>
    </div>
  );
}
