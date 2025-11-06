import { useState, useMemo } from "react";
import {
  pingHealth,
  processImages,
  applyBackgroundMean,
  computeFractions,
  downloadSelected,
} from "./lib/api";
import RoiCanvas from "./components/RoiCanvas";
import Histogram from "./components/Histogram";
import ResultsPanel from "./components/ResultsPanel";

function hexPngToDataUrl(hex) {
  const bytes = new Uint8Array(hex.match(/.{1,2}/g).map((b) => parseInt(b, 16)));
  const blob = new Blob([bytes], { type: "image/png" });
  return URL.createObjectURL(blob);
}

export default function App() {
  // uploads
  const [cerenkov, setCerenkov] = useState(null);
  const [dark, setDark] = useState(null);
  const [flat, setFlat] = useState(null); // .mat with fracMap
  const [bf, setBf] = useState(null);
  const [bin, setBin] = useState(1);

  // system
  const [ping, setPing] = useState(null);
  const [err, setErr] = useState("");

  // preview + histogram
  const [previewHex, setPreviewHex] = useState("");
  const [hist, setHist] = useState(null);
  const [serverWindow, setServerWindow] = useState({ low: 0, high: 255 });
  const [viewWindow, setViewWindow] = useState({ low: 0, high: 255 });
  const previewUrl = useMemo(
    () => (previewHex ? hexPngToDataUrl(previewHex) : ""),
    [previewHex]
  );

  // background ROI
  const [bgPoly, setBgPoly] = useState(null);

  // lane/band ROIs
  const [numLanes, setNumLanes] = useState(1);
  const [numBands, setNumBands] = useState(1);
  const [roiPolys, setRoiPolys] = useState([]); // lanes -> bands -> {points:[]}
  const [activeLane, setActiveLane] = useState(1);
  const [activeBand, setActiveBand] = useState(1);

  // results (fractions/sums)
  const [results, setResults] = useState(null);

  // downloads
  const [dlCerenkov, setDlCerenkov] = useState(true);
  const [dlBF, setDlBF] = useState(true);
  const [dlProcessed, setDlProcessed] = useState(true);
  const [dlProcessedRoi, setDlProcessedRoi] = useState(true);

  function ensureRoiShape(nL, nB) {
    setRoiPolys((old) => {
      let out = Array.isArray(old) ? old.map((r) => r.slice()) : [];
      if (out.length !== nL) out = Array.from({ length: nL }, () => []);
      for (let i = 0; i < nL; i++) {
        if (!Array.isArray(out[i]) || out[i].length !== nB) {
          out[i] = Array.from({ length: nB }, () => ({ points: [] }));
        } else {
          out[i] = out[i].map((cell) =>
            cell && Array.isArray(cell.points) ? cell : { points: [] }
          );
        }
      }
      return out;
    });
  }

  const onPing = async () => {
    setErr("");
    try {
      const j = await pingHealth();
      setPing(j);
    } catch (e) {
      setErr(e.message || String(e));
    }
  };

  const onUploadProcess = async () => {
    setErr("");
    setPing(null);
    setResults(null);
    if (!cerenkov || !dark || !flat || !bf) {
      setErr("Select Cerenkov, Dark, Flat (.mat), and Bright-field files.");
      return;
    }
    try {
      const j = await processImages({
        cerenkovFile: cerenkov,
        darkFile: dark,
        flatMatFile: flat, // IMPORTANT: .mat
        bfFile: bf,
        bin,
      });
      setPreviewHex(j.preview_png_hex);
      setHist(j.hist);
      setServerWindow(j.window);
      setViewWindow({ low: 0, high: 255 });
      setBgPoly(null);
      ensureRoiShape(numLanes, numBands);
    } catch (e) {
      setErr(e.message || String(e));
    }
  };

  const onDrawBackgroundFinish = ({ points }) => {
    setBgPoly(points);
  };

  const onApplyBackground = async () => {
    if (!bgPoly || bgPoly.length < 3) {
      setErr("Draw a background polygon first.");
      return;
    }
    try {
      const j = await applyBackgroundMean(bgPoly);
      setPreviewHex(j.preview_png_hex);
      setHist(j.hist);
      setServerWindow(j.window);
      setViewWindow({ low: 0, high: 255 });
    } catch (e) {
      setErr(e.message || String(e));
    }
  };

  const onDrawRoiFinish = ({ points }) => {
    const i = activeLane - 1,
      j = activeBand - 1;
    setRoiPolys((old) => {
      const out = old.map((row) => row.map((col) => ({ points: [...col.points] })));
      out[i][j] = { points };
      return out;
    });
  };

  const onRecompute = async () => {
    setErr("");
    setResults(null);
    // build rois payload with shape [num_lanes][num_bands] -> [[x,y],...]
    const rois = roiPolys.map((lane) => lane.map((b) => b.points || []));
    // quick validation to avoid 422
    if (rois.length !== numLanes || rois.some((r) => r.length !== numBands)) {
      setErr("ROI grid doesn’t match num_lanes/num_bands.");
      return;
    }
    try {
      const j = await computeFractions({
        num_lanes: numLanes,
        num_bands: numBands,
        rois,
        use_corrected4: true,
      });
      setResults(j);
    } catch (e) {
      setErr(e.message || String(e));
    }
  };

  const onDownloadCsv = () => {
    if (!results?.csv) return;
    const blob = new Blob([results.csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "results.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const onDownload = async () => {
    const which = [];
    if (dlCerenkov) which.push("cerenkov");
    if (dlBF) which.push("brightfield");
    if (dlProcessed) which.push("processed");
    if (dlProcessedRoi) which.push("processed_roi");
    if (which.length === 0) {
      setErr("Select at least one file to download.");
      return;
    }
    const rois = roiPolys.map((lane) => lane.map((b) => b.points || []));
    try {
      const blob = await downloadSelected({
        which,
        rois,
        window: viewWindow,
      });
      const a = document.createElement("a");
      const url = URL.createObjectURL(blob);
      a.href = url;
      a.download =
        which.length > 1
          ? "exports.zip"
          : which[0] === "processed_roi"
          ? "processed_ROI.tiff"
          : `${which[0]}.tiff`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setErr(e.message || String(e));
    }
  };

  return (
    <div
      style={{
        padding: 16,
        color: "#ddd",
        background: "#121212",
        minHeight: "100vh",
        fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
      }}
    >
      <h1 style={{ marginTop: 0 }}>TLC Analyzer</h1>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button onClick={onPing}>Ping API</button>
        {ping && <span style={{ color: "#9ef01a" }}>API OK</span>}
        {err && <span style={{ color: "salmon" }}>{err}</span>}
      </div>

      <div style={{ display: "grid", gap: 12, gridTemplateColumns: "1fr 1fr" }}>
        <div style={{ border: "1px solid #333", padding: 12, borderRadius: 8 }}>
          <h3>Upload & Prepare</h3>
          <label style={{ display: "block", marginBottom: 6 }}>
            Cerenkov (TIFF or BIN)
            <input
              type="file"
              onChange={(e) => setCerenkov(e.target.files?.[0] || null)}
              accept=".tif,.tiff,.bin"
            />
          </label>
          <label style={{ display: "block", marginBottom: 6 }}>
            Dark (TIFF or BIN)
            <input
              type="file"
              onChange={(e) => setDark(e.target.files?.[0] || null)}
              accept=".tif,.tiff,.bin"
            />
          </label>
          <label style={{ display: "block", marginBottom: 6 }}>
            Flat (.mat with fracMap)
            <input
              type="file"
              onChange={(e) => setFlat(e.target.files?.[0] || null)}
              accept=".mat"
            />
          </label>
          <label style={{ display: "block", marginBottom: 6 }}>
            Bright-field (TIFF or BIN)
            <input
              type="file"
              onChange={(e) => setBf(e.target.files?.[0] || null)}
              accept=".tif,.tiff,.bin"
            />
          </label>

          <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 6 }}>
            <label>
              Binning{" "}
              <input
                type="number"
                min="1"
                max="4"
                step="1"
                value={bin}
                onChange={(e) => setBin(parseInt(e.target.value || "1"))}
              />
            </label>
            <button onClick={onUploadProcess}>Prepare Preview</button>
          </div>

          <div style={{ marginTop: 12 }}>
            <Histogram
              hist={hist}
              windowLevel={viewWindow}
              onChange={(wl) =>
                setViewWindow({
                  low: Math.min(wl.low, wl.high - 1),
                  high: Math.max(wl.high, wl.low + 1),
                })
              }
            />
          </div>
        </div>

        <div style={{ border: "1px solid #333", padding: 12, borderRadius: 8 }}>
          <h3>Preview / Background ROI</h3>
          {!previewUrl && <div style={{ opacity: 0.7 }}>Upload and click “Prepare Preview”.</div>}
          {previewUrl && (
            <>
              <RoiCanvas
                imageUrl={previewUrl}
                mode="draw"
                windowLevel={viewWindow}
                polygons={
                  bgPoly ? [{ points: bgPoly, color: "rgba(180,255,64,0.7)", label: "BG" }] : []
                }
                onFinish={onDrawBackgroundFinish}
              />
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                <button onClick={onApplyBackground}>Apply Background</button>
                <button onClick={() => setBgPoly(null)}>Clear BG ROI</button>
              </div>
            </>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16, display: "grid", gap: 12, gridTemplateColumns: "1fr 1fr" }}>
        <div style={{ border: "1px solid #333", padding: 12, borderRadius: 8 }}>
          <h3>Lane/Band ROIs</h3>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <label>
              Lanes{" "}
              <input
                type="number"
                min="1"
                value={numLanes}
                onChange={(e) => {
                  const n = parseInt(e.target.value || "1");
                  setNumLanes(n);
                  ensureRoiShape(n, numBands);
                }}
              />
            </label>
            <label>
              Bands/lane{" "}
              <input
                type="number"
                min="1"
                value={numBands}
                onChange={(e) => {
                  const n = parseInt(e.target.value || "1");
                  setNumBands(n);
                  ensureRoiShape(numLanes, n);
                }}
              />
            </label>
            <label>
              Active Lane{" "}
              <input
                type="number"
                min="1"
                max={numLanes}
                value={activeLane}
                onChange={(e) => setActiveLane(parseInt(e.target.value || "1"))}
              />
            </label>
            <label>
              Active Band{" "}
              <input
                type="number"
                min="1"
                max={numBands}
                value={activeBand}
                onChange={(e) => setActiveBand(parseInt(e.target.value || "1"))}
              />
            </label>
            <button onClick={onRecompute}>Recompute Fractions</button>
          </div>

          {previewUrl && (
            <div style={{ marginTop: 8 }}>
              <RoiCanvas
                imageUrl={previewUrl}
                mode="draw"
                windowLevel={viewWindow}
                polygons={
                  // decorate saved polys with labels
                  roiPolys.flatMap((lane, i) =>
                    lane.map((b, j) =>
                      b?.points?.length >= 3
                        ? { points: b.points, label: `L${i + 1}B${j + 1}` }
                        : null
                    ).filter(Boolean)
                  )
                }
                onFinish={onDrawRoiFinish}
              />
            </div>
          )}
        </div>

        <div style={{ border: "1px solid #333", padding: 12, borderRadius: 8 }}>
          <h3>Results</h3>
          <ResultsPanel results={results} onDownloadCsv={onDownloadCsv} />

          <h3 style={{ marginTop: 16 }}>Download</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
            <label>
              <input
                type="checkbox"
                checked={dlCerenkov}
                onChange={(e) => setDlCerenkov(e.target.checked)}
              />{" "}
              Cerenkov.tiff
            </label>
            <label>
              <input
                type="checkbox"
                checked={dlBF}
                onChange={(e) => setDlBF(e.target.checked)}
              />{" "}
              Bright-field.tiff
            </label>
            <label>
              <input
                type="checkbox"
                checked={dlProcessed}
                onChange={(e) => setDlProcessed(e.target.checked)}
              />{" "}
              Processed.tiff
            </label>
            <label>
              <input
                type="checkbox"
                checked={dlProcessedRoi}
                onChange={(e) => setDlProcessedRoi(e.target.checked)}
              />{" "}
              Processed_ROI.tiff
            </label>
          </div>
          <div style={{ marginTop: 8 }}>
            <button onClick={onDownload}>Download Selected</button>
          </div>
          <div style={{ opacity: 0.7, fontSize: 12, marginTop: 6 }}>
            If multiple are selected, you’ll get a single ZIP.
          </div>
        </div>
      </div>
    </div>
  );
}
