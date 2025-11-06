from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Literal, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import tifffile as tiff
from scipy.ndimage import median_filter
from skimage import measure, filters, morphology, draw as skdraw
from scipy.io import loadmat
import zipfile
import json
import os

app = FastAPI(title="TLC Analyzer API", version="0.3.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://thilinahwe.github.io"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------
# In-memory session cache (simple)
# ------------------------------
SESSION: Dict[str, Any] = {
    "inputs": {},        # original filenames
    "data16": None,      # uint16 cerenkov
    "dark16": None,      # uint16 dark
    "flat32": None,      # float fracMap
    "bf16": None,        # uint16 brightfield
    "corrected4": None,  # pre-background (float)
    "corrected5": None,  # post-background (float)
    "mean_bg": 0.0,
    "bin": 1
}

def _strip_ext(name: str) -> str:
    base = os.path.basename(name or "image")
    if "." in base:
        return ".".join(base.split(".")[:-1]) or base
    return base

def _read_tiff_bytes_to_u16(b: bytes) -> np.ndarray:
    with BytesIO(b) as bio:
        arr = tiff.imread(bio)
    if arr.dtype == np.uint16:
        return arr
    if arr.dtype == np.uint8:
        return (arr.astype(np.uint16) * 257)
    return arr.astype(np.uint16)

def _read_bin_u16(b: bytes, shape=(3008, 3008)) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint16)
    need = shape[0] * shape[1]
    if arr.size != need:
        raise ValueError(f"BIN size {arr.size} != expected {need}")
    return arr.reshape(shape)

def _bin_reduce_sum(img: np.ndarray, binf: int) -> np.ndarray:
    if binf == 1:
        return img
    h, w = img.shape
    if binf == 3:
        h = (h // 3) * 3
        w = (w // 3) * 3
        img = img[:h, :w]
    new = np.zeros((h // binf, w // binf), dtype=img.dtype)
    for i in range(binf):
        for j in range(binf):
            new += img[i::binf, j::binf]
    return new

def _apply_pipeline(data16, dark16, flat32, binf: int) -> np.ndarray:
    data2 = _bin_reduce_sum(data16, binf).astype(np.float64)
    dark2 = _bin_reduce_sum(dark16, binf).astype(np.float64)
    flat2 = _bin_reduce_sum(flat32, binf).astype(np.float64)

    corrected = data2 - dark2                # dark subtraction (multiplier = 1)
    corrected2 = corrected * flat2           # flat/vignetting correction
    corrected3 = median_filter(corrected2, size=3)  # median 3x3
    corrected4 = np.rot90(corrected3, k=2)   # rotate 180°
    return corrected4

def _window_u8(img_float: np.ndarray, lo: float, hi: float) -> np.ndarray:
    lo = max(float(np.min(img_float)), float(lo))
    hi = max(lo + 1e-6, float(hi))
    v = (img_float - lo) * (255.0 / (hi - lo))
    return np.clip(v, 0, 255).astype(np.uint8)

def _poly_mask(h: int, w: int, pts: List[List[float]]) -> np.ndarray:
    if not pts:
        return np.zeros((h, w), dtype=bool)
    rr, cc = skdraw.polygon([p[1] for p in pts], [p[0] for p in pts], shape=(h, w))
    m = np.zeros((h, w), dtype=bool)
    m[rr, cc] = True
    return m

def _draw_polys_on_u8(base: np.ndarray, lanes: List[List[List[List[float]]]]) -> Image.Image:
    img = Image.fromarray(base, mode="L").convert("RGB")
    drw = ImageDraw.Draw(img)
    colors = ["#ff5252", "#ffd166", "#06d6a0", "#4cc9f0", "#f72585", "#b2ff59"]
    for i, lane in enumerate(lanes, start=1):
        for j, poly in enumerate(lane, start=1):
            if not poly:
                continue
            col = colors[(i + j) % len(colors)]
            xy = [(float(x), float(y)) for x, y in poly]
            drw.polygon(xy, outline=col)
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            drw.text((cx + 4, cy + 4), f"L{i}B{j}", fill=col)
    return img

@app.get("/health")
def health():
    return {"ok": True, "service": "tlc-analyzer-api"}

def _load_flat_to_float32(file: UploadFile, raw: bytes) -> Optional[np.ndarray]:
    """
    Accept .mat (keys like 'fracMap', 'flat', 'Flat', 'MasterFlat'), or TIFF.
    Return float32 2D array, or None if not parsable.
    """
    name = (file.filename or "").lower()
    # .mat
    if name.endswith(".mat"):
        try:
            mat = loadmat(BytesIO(raw))
            # common key candidates
            for k in ["fracMap", "flat", "Flat", "MasterFlat", "flatMap", "FlatMap"]:
                if k in mat:
                    arr = np.array(mat[k])
                    if arr.ndim > 2:
                        arr = np.squeeze(arr)
                    arr = arr.astype(np.float32)
                    return arr
            # couldn’t find key
            return None
        except Exception:
            return None
    # TIFF
    if name.endswith((".tif", ".tiff")):
        try:
            arr = _read_tiff_bytes_to_u16(raw).astype(np.float32)
            return arr
        except Exception:
            return None
    # JSON with "fracMap"
    try:
        j = json.loads(raw.decode("utf-8"))
        if "fracMap" in j:
            arr = np.array(j["fracMap"], dtype=np.float32)
            return arr
    except Exception:
        pass
    return None

@app.post("/process")
async def process_images(
    cerenkov: UploadFile = File(..., description="Cerenkov image .tiff or .bin"),
    dark: UploadFile = File(..., description="Dark image .tiff or .bin"),
    flat: UploadFile = File(..., description=".mat with fracMap, .tiff, or JSON with key 'fracMap'"),
    bf: UploadFile = File(..., description="Bright-field .tiff or .bin"),
    bin: int = Form(1),
):
    """
    Upload + MATLAB-like pipeline → corrected4 (pre-background).
    Returns: 8-bit preview + histogram + window.
    """
    # record names
    SESSION["inputs"] = {
        "cerenkov_name": cerenkov.filename,
        "dark_name": dark.filename,
        "flat_name": flat.filename,
        "bf_name": bf.filename,
    }
    SESSION["bin"] = int(bin)

    # read image bytes
    cbytes = await cerenkov.read()
    dbytes = await dark.read()
    bbytes = await bf.read()
    fbytes = await flat.read()

    # decode images
    def read_img(up: UploadFile, raw: bytes) -> np.ndarray:
        fname = (up.filename or "").lower()
        if fname.endswith(".bin"):
            return _read_bin_u16(raw)          # expects 3008x3008
        return _read_tiff_bytes_to_u16(raw)

    try:
        data16 = read_img(cerenkov, cbytes)
        dark16 = read_img(dark, dbytes)
        bf16   = read_img(bf, bbytes)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Failed to read image(s): {e}"})

    flat32 = _load_flat_to_float32(flat, fbytes)
    if flat32 is None:
        return JSONResponse(status_code=400, content={
            "error": "Could not parse 'flat'. Supply a .mat with key like 'fracMap' (or 'flat','MasterFlat'), a .tiff, or JSON with 'fracMap'."
        })

    # stash
    SESSION["data16"] = data16
    SESSION["dark16"] = dark16
    SESSION["flat32"] = flat32
    SESSION["bf16"]   = bf16

    # pipeline
    corrected4 = _apply_pipeline(data16, dark16, flat32, SESSION["bin"])
    SESSION["corrected4"] = corrected4
    SESSION["corrected5"] = corrected4.copy()
    SESSION["mean_bg"] = 0.0

    lo, hi = float(corrected4.min()), float(corrected4.max())
    prev = _window_u8(corrected4, lo, hi)
    hist, _ = np.histogram(prev, bins=256, range=(0, 255))
    buf = BytesIO(); Image.fromarray(prev).save(buf, format="PNG"); buf.seek(0)

    return {
        "preview_png_hex": buf.getvalue().hex(),
        "shape": corrected4.shape,
        "window": {"low": lo, "high": hi},
        "hist": hist.tolist()
    }

@app.post("/background-mean")
async def background_mean(payload: dict):
    if SESSION.get("corrected4") is None:
        return JSONResponse(status_code=400, content={"error": "No session. Upload via /process first."})

    corrected4 = SESSION["corrected4"]
    h, w = corrected4.shape
    pts = payload.get("polygon", [])
    if not pts:
        return JSONResponse(status_code=400, content={"error": "No polygon points."})

    mask = _poly_mask(h, w, pts)
    mean_bg = float(np.mean(corrected4[mask]))
    SESSION["mean_bg"] = mean_bg
    corrected5 = corrected4 - mean_bg
    SESSION["corrected5"] = corrected5

    lo, hi = float(corrected5.min()), float(corrected5.max())
    prev = _window_u8(corrected5, lo, hi)
    hist, _ = np.histogram(prev, bins=256, range=(0, 255))
    buf = BytesIO(); Image.fromarray(prev).save(buf, format="PNG"); buf.seek(0)

    return {
        "mean_bg": mean_bg,
        "preview_png_hex": buf.getvalue().hex(),
        "window": {"low": lo, "high": hi},
        "hist": hist.tolist(),
    }

@app.post("/detect")
async def detect_rois(payload: dict):
    if SESSION.get("corrected4") is None:
        return JSONResponse(status_code=400, content={"error": "No session."})

    mode: Literal["circle", "rectangle"] = payload.get("mode", "circle")
    use_c5 = bool(payload.get("use_corrected5", True))
    img = SESSION["corrected5"] if use_c5 and SESSION["corrected5"] is not None else SESSION["corrected4"]
    h, w = img.shape

    roi_mask = np.ones((h, w), dtype=bool)
    lane_poly = payload.get("lane_polygon")
    if lane_poly:
        roi_mask = _poly_mask(h, w, lane_poly)

    lo, hi = img.min(), img.max()
    view = np.clip((img - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    view[~roi_mask] = 0

    try:
        th = filters.threshold_otsu(view[roi_mask])
    except ValueError:
        th = 0
    bw = view > th
    min_area = int(payload.get("params", {}).get("min_area", 50))
    bw = morphology.remove_small_objects(bw, min_size=min_area)
    lab = measure.label(bw)
    props = measure.regionprops(lab)

    results = []
    for p in props:
        y0, x0, y1, x1 = p.bbox
        area = p.area
        if area <= 0:
            continue

        if mode == "circle":
            per = p.perimeter if p.perimeter > 0 else 1.0
            circ = 4.0 * np.pi * area / (per * per)
            if circ < float(payload.get("params", {}).get("min_circularity", 0.4)):
                continue
            r = float(np.sqrt(area / np.pi))
            cx, cy = float(p.centroid[1]), float(p.centroid[0])
            results.append({"type": "circle", "cx": cx, "cy": cy, "r": r, "area": int(area), "circularity": float(circ)})
        else:
            ww = x1 - x0; hh = y1 - y0
            if ww <= 0 or hh <= 0:
                continue
            ar = ww / float(hh)
            min_ar = float(payload.get("params", {}).get("min_aspect", 0.2))
            max_ar = float(payload.get("params", {}).get("max_aspect", 10.0))
            if not (min_ar <= ar <= max_ar):
                continue
            results.append({"type": "rect", "x": float(x0), "y": float(y0), "w": float(ww), "h": float(hh), "area": int(area), "aspect": float(ar)})

    return {"count": len(results), "rois": results}

@app.post("/roi/fractions")
async def roi_fractions(payload: dict):
    if SESSION.get("corrected4") is None:
        return JSONResponse(status_code=400, content={"error": "No session."})

    num_lanes = int(payload.get("num_lanes", 1))
    num_bands = int(payload.get("num_bands", 1))
    rois = payload.get("rois", [])
    use_c4 = bool(payload.get("use_corrected4", True))

    img = SESSION["corrected4"] if use_c4 else (SESSION["corrected5"] if SESSION["corrected5"] is not None else SESSION["corrected4"])
    h, w = img.shape

    sums = np.zeros((num_lanes, num_bands), dtype=np.float64)
    for i in range(num_lanes):
        for j in range(num_bands):
            try:
                pts = rois[i][j]
            except Exception:
                pts = []
            if not pts:
                continue
            mask = _poly_mask(h, w, pts)
            sums[i, j] = float(np.sum(img[mask]))

    lane_sums = np.sum(sums, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        fractions = np.divide(sums, lane_sums, where=lane_sums > 0)

    header_frac = [f"Band {k+1} (frac)" for k in range(num_bands)]
    header_sig  = [f"Band {k+1} (signal)" for k in range(num_bands)]
    rows = []
    for i in range(num_lanes):
        row = {"Lane": i + 1}
        for j in range(num_bands):
            row[header_frac[j]] = float(fractions[i, j]) if lane_sums[i, 0] > 0 else 0.0
        for j in range(num_bands):
            row[header_sig[j]] = float(sums[i, j])
        rows.append(row)

    csv_cols = ["Lane"] + header_frac + header_sig
    out = [",".join(csv_cols)]
    for i in range(num_lanes):
        vals = [str(i + 1)]
        vals += [f"{(fractions[i, j] if lane_sums[i, 0] > 0 else 0.0):.6f}" for j in range(num_bands)]
        vals += [f"{sums[i, j]:.3f}" for j in range(num_bands)]
        out.append(",".join(vals))
    csv_str = "\n".join(out)

    return {"table": rows, "csv": csv_str}

@app.post("/download")
async def download_tiffs(payload: dict):
    which = payload.get("which", [])
    if not which:
        return JSONResponse(status_code=400, content={"error": "Specify 'which' list."})
    if SESSION.get("corrected4") is None:
        return JSONResponse(status_code=400, content={"error": "No session."})

    c_base = _strip_ext(SESSION["inputs"].get("cerenkov_name"))
    b_base = _strip_ext(SESSION["inputs"].get("bf_name"))

    files: List[tuple[str, bytes]] = []

    def to_tiff_bytes_u16(arr_u16: np.ndarray) -> bytes:
        bio = BytesIO()
        tiff.imwrite(bio, arr_u16, dtype=np.uint16)
        return bio.getvalue()

    def to_tiff_bytes_u8(arr_u8: np.ndarray, mode="L") -> bytes:
        bio = BytesIO()
        Image.fromarray(arr_u8, mode=mode).save(bio, format="TIFF")
        return bio.getvalue()

    corrected5 = SESSION["corrected5"] if SESSION["corrected5"] is not None else SESSION["corrected4"]
    proc_u16 = np.clip(corrected5, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    if "cerenkov" in which:
        files.append((f"{c_base}.tiff", to_tiff_bytes_u16(SESSION["data16"].astype(np.uint16))))
    if "brightfield" in which:
        files.append((f"{b_base}.tiff", to_tiff_bytes_u16(SESSION["bf16"].astype(np.uint16))))
    if "processed" in which:
        files.append((f"{c_base}_processed.tiff", to_tiff_bytes_u16(proc_u16)))
    if "processed_roi" in which:
        wl = payload.get("window", {})
        lo = float(wl.get("low", float(proc_u16.min())))
        hi = float(wl.get("high", float(proc_u16.max())))
        view8 = _window_u8(proc_u16.astype(np.float64), lo, hi)
        rois = payload.get("rois", [])
        overlay_img = _draw_polys_on_u8(view8, rois)
        bio = BytesIO(); overlay_img.save(bio, format="TIFF"); bio.seek(0)
        files.append((f"{c_base}_processed_ROI.tiff", bio.getvalue()))

    if len(files) == 1:
        name, data = files[0]
        return StreamingResponse(BytesIO(data), media_type="image/tiff",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})
    zbuf = BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    zbuf.seek(0)
    return StreamingResponse(zbuf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{c_base}_exports.zip"'})
