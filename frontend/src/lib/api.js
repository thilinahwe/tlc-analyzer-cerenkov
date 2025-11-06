const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function pingHealth() {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function processImages({ cerenkovFile, darkFile, flatMatFile, bfFile, bin }) {
  const fd = new FormData();
  fd.append("cerenkov", cerenkovFile);
  fd.append("dark", darkFile);
  fd.append("flat", flatMatFile);   // MUST be .mat with 'fracMap'
  fd.append("bf", bfFile);
  fd.append("bin", String(bin || 1));

  const r = await fetch(`${API_BASE}/process`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function applyBackgroundMean(polygon) {
  const r = await fetch(`${API_BASE}/background-mean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ polygon }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function detectROIs({ mode = "circle", use_corrected5 = true, lane_polygon = null, params = {} }) {
  const r = await fetch(`${API_BASE}/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, use_corrected5, lane_polygon, params }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function computeFractions({ num_lanes, num_bands, rois, use_corrected4 = true }) {
  const r = await fetch(`${API_BASE}/roi/fractions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_lanes, num_bands, rois, use_corrected4 }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function downloadSelected({ which, rois, window }) {
  const r = await fetch(`${API_BASE}/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ which, rois, window }),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const blob = await r.blob();
  return blob;
}
