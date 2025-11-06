import React from "react";

export default function ResultsPanel({ results, onDownloadCsv }) {
  if (!results) {
    return (
      <div style={{ opacity: 0.7 }}>No results yet. Draw ROIs and click “Recompute Fractions”.</div>
    );
  }

  const rows = results.table || [];
  if (rows.length === 0) {
    return <div style={{ opacity: 0.7 }}>No rows to display.</div>;
  }

  // Build column order from first row keys (ensure "Lane" first)
  const allKeys = Object.keys(rows[0]);
  const laneIdx = allKeys.indexOf("Lane");
  if (laneIdx > -1) {
    allKeys.splice(laneIdx, 1);
  }
  const columns = ["Lane", ...allKeys];

  return (
    <div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <h4 style={{ margin: 0 }}>Results</h4>
        <button onClick={onDownloadCsv}>Download CSV</button>
      </div>

      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            background: "#181818",
            color: "#ddd",
            border: "1px solid #333",
          }}
        >
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  style={{
                    textAlign: "left",
                    padding: "6px 8px",
                    borderBottom: "1px solid #333",
                    position: "sticky",
                    top: 0,
                    background: "#202020",
                  }}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => (
              <tr key={idx}>
                {columns.map((c) => (
                  <td key={c} style={{ padding: "6px 8px", borderBottom: "1px solid #2a2a2a" }}>
                    {typeof r[c] === "number" ? Number(r[c]).toString() : r[c]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>
        Columns “Band k (frac)” match MATLAB’s lane-normalized fractions; “Band k (signal)” are raw sums.
      </div>
    </div>
  );
}
