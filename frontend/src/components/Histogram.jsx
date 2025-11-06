export default function Histogram({ hist, windowLevel, onChange }) {
  const low = windowLevel?.low ?? 0;
  const high = windowLevel?.high ?? 255;

  return (
    <div style={{ padding: 8, border: "1px solid #444", borderRadius: 6 }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>Histogram / Window</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <label>
          Low
          <input
            type="range" min="0" max="255" value={Math.round(low)}
            onChange={(e) => onChange({ low: Number(e.target.value), high })}
          />
          <span style={{ marginLeft: 8 }}>{Math.round(low)}</span>
        </label>
        <label>
          High
          <input
            type="range" min="0" max="255" value={Math.round(high)}
            onChange={(e) => onChange({ low, high: Number(e.target.value) })}
          />
          <span style={{ marginLeft: 8 }}>{Math.round(high)}</span>
        </label>
      </div>
      {hist && (
        <pre style={{ fontSize: 10, maxHeight: 120, overflow: "auto", background: "#111", padding: 6 }}>
          {JSON.stringify(hist.slice(0, 64))} ... ({hist.length} bins)
        </pre>
      )}
    </div>
  );
}
