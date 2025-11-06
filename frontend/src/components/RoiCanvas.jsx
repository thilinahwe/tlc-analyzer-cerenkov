import { useEffect, useRef, useState } from "react";

/**
 * Props:
 * - imageUrl (base64 PNG or URL)
 * - mode: "draw" | "view"
 * - polygons: [{ points: [[x,y],...], label?: string, color?: string }]
 * - onFinish?: (poly) => void  // called with {points}
 * - showAxes?: boolean
 * - windowLevel?: { low: number, high: number }  // 0..255 display window
 */
export default function RoiCanvas({
  imageUrl,
  mode = "view",
  polygons = [],
  onFinish,
  showAxes = false,
  windowLevel = { low: 0, high: 255 },
}) {
  const canvasRef = useRef(null);

  // Raw image pixels (8-bit) captured once from the source PNG.
  const [imgPixels, setImgPixels] = useState(null); // Uint8ClampedArray RGBA
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  const [drawing, setDrawing] = useState(false);
  const [current, setCurrent] = useState([]); // [[x,y],...]

  useEffect(() => {
    if (!imageUrl) {
      setImgPixels(null);
      setImgSize({ w: 0, h: 0 });
      return;
    }
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      const w = im.width;
      const h = im.height;
      const off = document.createElement("canvas");
      off.width = w; off.height = h;
      const octx = off.getContext("2d", { willReadFrequently: true });
      octx.drawImage(im, 0, 0);
      const data = octx.getImageData(0, 0, w, h); // RGBA
      setImgPixels(data.data);
      setImgSize({ w, h });
      requestAnimationFrame(() => paint());
    };
    im.src = imageUrl;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageUrl]);

  useEffect(() => {
    paint();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [windowLevel, polygons, current]);

  function paint() {
    const canvas = canvasRef.current;
    if (!canvas || !imgPixels) return;
    const { w, h } = imgSize;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
    const ctx = canvas.getContext("2d");

    // 1) windowed grayscale into new ImageData
    const out = ctx.createImageData(w, h);
    const lo = Math.max(0, Math.min(255, windowLevel.low ?? 0));
    const hi = Math.max(lo + 1, Math.min(255, windowLevel.high ?? 255));
    const scale = 255 / (hi - lo);

    for (let i = 0; i < imgPixels.length; i += 4) {
      let v = imgPixels[i]; // R channel
      let m = (v - lo) * scale;
      if (m < 0) m = 0; else if (m > 255) m = 255;
      const u = m | 0;
      out.data[i] = u;
      out.data[i + 1] = u;
      out.data[i + 2] = u;
      out.data[i + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);

    if (showAxes) {
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 100) {
        ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, h); ctx.stroke();
      }
      for (let y = 0; y < h; y += 100) {
        ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(w, y + 0.5); ctx.stroke();
      }
    }

    // saved polygons
    for (const poly of polygons) {
      drawPoly(ctx, poly.points, poly.color || "rgba(255,80,80,0.7)");
      if (poly.label) {
        const [cx, cy] = centroid(poly.points);
        drawLabel(ctx, poly.label, cx, cy);
      }
    }

    // live stroke
    if (current.length > 1) {
      drawPoly(ctx, current, "rgba(64,180,255,0.75)");
    }
  }

  function drawPoly(ctx, pts, stroke = "rgba(255,80,80,0.7)") {
    if (!pts || pts.length < 2) return;
    ctx.lineWidth = 2;
    ctx.strokeStyle = stroke;
    let fill = "rgba(255,80,80,0.25)";
    try {
      fill = stroke.replace(/rgba\(([^)]+),\s*([0-9.]+)\)/, "rgba($1,0.25)");
    } catch {}
    ctx.fillStyle = fill;

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  function drawLabel(ctx, text, x, y) {
    ctx.font = "14px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "yellow";
    ctx.strokeStyle = "rgba(0,0,0,0.7)";
    ctx.lineWidth = 3;
    ctx.strokeText(text, x + 4, y + 4);
    ctx.fillText(text, x + 4, y + 4);
  }

  function centroid(pts) {
    let sx = 0, sy = 0;
    for (const [x, y] of pts) { sx += x; sy += y; }
    return [sx / pts.length, sy / pts.length];
  }

  function onMouseDown(e) {
    if (mode !== "draw" || !imgPixels) return;
    const { x, y } = toLocal(e);
    setDrawing(true);
    setCurrent([[x, y]]);
  }

  function onMouseMove(e) {
    if (!drawing || mode !== "draw" || !imgPixels) return;
    const { x, y } = toLocal(e);
    setCurrent((c) => {
      const last = c[c.length - 1];
      if (last && Math.hypot(last[0] - x, last[1] - y) < 1) return c;
      return [...c, [x, y]];
    });
  }

  function onMouseUp() {
    if (mode !== "draw" || !drawing) return;
    setDrawing(false);
    if (onFinish && current.length > 2) onFinish({ points: current });
    setCurrent([]);
  }

  function toLocal(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    let x = (e.clientX - rect.left) * sx;
    let y = (e.clientY - rect.top) * sy;
    x = Math.max(0, Math.min(canvas.width, x));
    y = Math.max(0, Math.min(canvas.height, y));
    return { x, y };
  }

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        height: "auto",
        border: "1px solid #333",
        borderRadius: 6,
        cursor: mode === "draw" ? "crosshair" : "default",
        background: "#000",
        display: imgPixels ? "block" : "none",
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
    />
  );
}
