import React, { useEffect, useMemo, useRef, useState } from "react";
import * as faceapi from "face-api.js";
import "./App.css";

type ExprMap = Record<string, number>;
type Theme = "light" | "dark";
type Side = "right" | "left" | "top" | "bottom";
type Anchor = { side: Side; x: number; y: number };

function isMobileUA() {
  const ua = navigator.userAgent || "";
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
}

function topK(expr: ExprMap, k: number) {
  return Object.entries(expr).sort((a, b) => b[1] - a[1]).slice(0, k);
}

function emaBlend(prev: ExprMap | null, next: ExprMap, alpha: number) {
  if (!prev) return next;
  const out: ExprMap = {};
  const keys = new Set([...Object.keys(prev), ...Object.keys(next)]);
  keys.forEach((key) => {
    const p = prev[key] ?? 0;
    const n = next[key] ?? 0;
    out[key] = p * (1 - alpha) + n * alpha;
  });
  return out;
}

function reweightForDisplay(raw: ExprMap) {
  const neutralKey = "neutral";
  const boost = 1.25;
  const neutralMul = 0.6;

  const out: ExprMap = {};
  let sum = 0;

  for (const [k, v] of Object.entries(raw)) {
    const m = k === neutralKey ? neutralMul : boost;
    const nv = Math.max(0, v * m);
    out[k] = nv;
    sum += nv;
  }

  if (sum <= 0) return raw;
  for (const k of Object.keys(out)) out[k] = out[k] / sum;
  return out;
}

function clamp(val: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, val));
}

function computeSidePos(
  side: Side,
  box: { x: number; y: number; w: number; h: number },
  labelW: number,
  labelH: number,
  pad: number
) {
  if (side === "right") return { x: box.x + box.w + pad, y: box.y };
  if (side === "left") return { x: box.x - labelW - pad, y: box.y };
  if (side === "top") return { x: box.x, y: box.y - labelH - pad };
  return { x: box.x, y: box.y + box.h + pad };
}

function fitsWithMargin(
  x: number,
  y: number,
  labelW: number,
  labelH: number,
  vw: number,
  vh: number,
  margin: number
) {
  return x >= margin && y >= margin && x + labelW <= vw - margin && y + labelH <= vh - margin;
}

function decideAnchor(
  prev: Anchor | null,
  box: { x: number; y: number; w: number; h: number },
  labelW: number,
  labelH: number,
  vw: number,
  vh: number
): Anchor {
  const pad = 10;
  const strictMargin = 24;
  const keepMargin = 8;

  if (prev) {
    const p = computeSidePos(prev.side, box, labelW, labelH, pad);
    if (fitsWithMargin(p.x, p.y, labelW, labelH, vw, vh, keepMargin)) {
      return { side: prev.side, x: clamp(p.x, 0, vw - labelW), y: clamp(p.y, 0, vh - labelH) };
    }
  }

  const rightPos = computeSidePos("right", box, labelW, labelH, pad);
  if (fitsWithMargin(rightPos.x, rightPos.y, labelW, labelH, vw, vh, strictMargin)) {
    return { side: "right", x: clamp(rightPos.x, 0, vw - labelW), y: clamp(rightPos.y, 0, vh - labelH) };
  }

  const leftPos = computeSidePos("left", box, labelW, labelH, pad);
  if (fitsWithMargin(leftPos.x, leftPos.y, labelW, labelH, vw, vh, strictMargin)) {
    return { side: "left", x: clamp(leftPos.x, 0, vw - labelW), y: clamp(leftPos.y, 0, vh - labelH) };
  }

  const bottomPos = computeSidePos("bottom", box, labelW, labelH, pad);
  if (fitsWithMargin(bottomPos.x, bottomPos.y, labelW, labelH, vw, vh, 16)) {
    return { side: "bottom", x: clamp(bottomPos.x, 0, vw - labelW), y: clamp(bottomPos.y, 0, vh - labelH) };
  }

  const topPos = computeSidePos("top", box, labelW, labelH, pad);
  if (fitsWithMargin(topPos.x, topPos.y, labelW, labelH, vw, vh, 16)) {
    return { side: "top", x: clamp(topPos.x, 0, vw - labelW), y: clamp(topPos.y, 0, vh - labelH) };
  }

  return { side: "right", x: clamp(rightPos.x, 0, vw - labelW), y: clamp(rightPos.y, 0, vh - labelH) };
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [theme, setTheme] = useState<Theme>("light");
  const [flipX, setFlipX] = useState(true);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState("Loading models...");
  const [modelLoadMs, setModelLoadMs] = useState<number | null>(null);

  const [renderFps, setRenderFps] = useState(0);
  const [inferHz, setInferHz] = useState(0);

  const smoothedFacesRef = useRef<Array<ExprMap | null>>([]);
  const anchorFacesRef = useRef<Array<Anchor | null>>([]);

  const freakyUntilRef = useRef<number>(0);
  const freakyComboUntilRef = useRef<number>(0);

  const mobile = useMemo(() => isMobileUA(), []);

  const smoothingAlpha = 0.25;
  const inferEveryMs = 100;

  const boxOffsetY = -30;
  const boxInsetX = 30;

  const freakyCooldownMs = 4000;
  const comboWindowMs = 2000;
  const angryThresh = 0.30;
  const happyThresh = 0.30;
  const triggerChancePerTick = 0.20;

  useEffect(() => {
    (async () => {
      try {
        const t0 = performance.now();
        await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
        await faceapi.nets.faceExpressionNet.loadFromUri("/models");
        const t1 = performance.now();
        setModelLoadMs(Math.round(t1 - t0));
        setReady(true);
        setStatus("Ready. Click Start Camera.");
      } catch (e) {
        console.error(e);
        setStatus("Model load failed. Check public/models files.");
      }
    })();
  }, []);

  async function startCamera() {
    if (!ready) return;
    try {
      setStatus("Requesting camera permission...");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();

      smoothedFacesRef.current = [];
      anchorFacesRef.current = [];
      freakyUntilRef.current = 0;
      freakyComboUntilRef.current = 0;

      setRunning(true);
      setStatus("Running (on-device).");
    } catch (e) {
      console.error(e);
      setStatus("Camera permission denied or unavailable.");
    }
  }

  function clearCanvas() {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, c.width, c.height);
  }

  function stopCamera() {
    const v = videoRef.current;
    if (v?.srcObject) {
      (v.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
    smoothedFacesRef.current = [];
    anchorFacesRef.current = [];
    freakyUntilRef.current = 0;
    freakyComboUntilRef.current = 0;
    clearCanvas();
    setRunning(false);
    setStatus("Stopped.");
  }

  useEffect(() => {
    if (!running) return;

    const video = videoRef.current!;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;

    let rafId = 0;
    let lastInferAt = 0;
    let busy = false;

    let frames = 0;
    let lastFpsTick = performance.now();

    let infers = 0;
    let lastInferTick = performance.now();

    const detectorOptions = new faceapi.TinyFaceDetectorOptions({
      inputSize: 224,
      scoreThreshold: 0.4,
    });

    const loop = () => {
      rafId = requestAnimationFrame(loop);

      frames++;
      const now = performance.now();
      if (now - lastFpsTick >= 500) {
        setRenderFps(Math.round((frames * 1000) / (now - lastFpsTick)));
        frames = 0;
        lastFpsTick = now;
      }

      if (now - lastInferAt < inferEveryMs) return;
      if (busy) return;
      busy = true;
      lastInferAt = now;

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) {
        busy = false;
        return;
      }

      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }

      (async () => {
        try {
          const dets = await faceapi.detectAllFaces(video, detectorOptions).withFaceExpressions();

          infers++;
          const t = performance.now();
          if (t - lastInferTick >= 1000) {
            setInferHz(Math.round((infers * 1000) / (t - lastInferTick)));
            infers = 0;
            lastInferTick = t;
          }

          dets.sort((a, b) => a.detection.box.x - b.detection.box.x);

          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const nowMs = performance.now();

          if (dets.length === 0) {
            const inFreaky = nowMs < freakyUntilRef.current;
            if (inFreaky) {
              ctx.font = "34px sans-serif";
              const text = "FREAKY";
              const pad = 14;
              const w = ctx.measureText(text).width + pad * 2;
              const h = 52;
              ctx.fillStyle = "rgba(0,0,0,0.65)";
              ctx.fillRect(12, 12, w, h);
              ctx.fillStyle = "white";
              ctx.fillText(text, 12 + pad, 12 + 38);
            } else {
              ctx.font = "18px sans-serif";
              ctx.fillStyle = "rgba(0,0,0,0.65)";
              ctx.fillRect(12, 12, 220, 34);
              ctx.fillStyle = "white";
              ctx.fillText("No face detected", 20, 35);
            }
            smoothedFacesRef.current = [];
            anchorFacesRef.current = [];
            return;
          }

          let sawCombo = false;
          for (const d of dets) {
            const raw: ExprMap = d.expressions as any;
            const angry = raw["angry"] ?? 0;
            const happy = raw["happy"] ?? 0;
            if (angry >= angryThresh && happy >= happyThresh) {
              sawCombo = true;
              break;
            }
          }

          if (sawCombo) {
            freakyComboUntilRef.current = nowMs + comboWindowMs;
          }

          const inCooldown = nowMs < freakyUntilRef.current;
          const inComboWindow = nowMs < freakyComboUntilRef.current;

          if (!inCooldown && inComboWindow) {
            if (Math.random() < triggerChancePerTick) {
              freakyUntilRef.current = nowMs + freakyCooldownMs;
            }
          }

          const showFreaky = nowMs < freakyUntilRef.current;

          const nextSmoothed: Array<ExprMap | null> = [];
          const nextAnchors: Array<Anchor | null> = [];

          for (let i = 0; i < dets.length; i++) {
            const det = dets[i];
            const raw: ExprMap = det.expressions as any;
            const b = det.detection.box;

            let x = b.x + boxInsetX;
            let y = b.y + boxOffsetY;
            let w = b.width - 2 * boxInsetX;
            let h = b.height;

            w = Math.max(1, w);
            h = Math.max(1, h);

            x = clamp(x, 0, canvas.width - w);
            y = clamp(y, 0, canvas.height - h);

            const xDraw = flipX ? canvas.width - (x + w) : x;

            ctx.strokeStyle = "lime";
            ctx.lineWidth = 3;
            ctx.strokeRect(xDraw, y, w, h);

            if (showFreaky) {
              const text = "FREAKY";
              ctx.font = "34px sans-serif";
              const pad = 14;
              const lineH = 38;
              const labelW = ctx.measureText(text).width + pad * 2;
              const labelH = lineH + pad * 2;

              const prevAnchor = anchorFacesRef.current[i] ?? null;
              const anchor = decideAnchor(prevAnchor, { x: xDraw, y, w, h }, labelW, labelH, canvas.width, canvas.height);
              nextAnchors[i] = anchor;

              ctx.fillStyle = "rgba(0,0,0,0.65)";
              ctx.fillRect(anchor.x, anchor.y, labelW, labelH);

              ctx.fillStyle = "white";
              ctx.fillText(text, anchor.x + pad, anchor.y + pad + lineH - 10);

              nextSmoothed[i] = smoothedFacesRef.current[i] ?? null;
              continue;
            }

            const displayExpr = reweightForDisplay(raw);
            const prevSm = smoothedFacesRef.current[i] ?? null;
            const blended = emaBlend(prevSm, displayExpr, smoothingAlpha);
            nextSmoothed[i] = blended;

            const top3 = topK(blended, 3);
            const lines = top3.map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`);

            ctx.font = "18px sans-serif";
            const pad = 10;
            const lineH = 22;
            const labelW = Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2;
            const labelH = lines.length * lineH + pad * 2;

            const prevAnchor = anchorFacesRef.current[i] ?? null;
            const anchor = decideAnchor(prevAnchor, { x: xDraw, y, w, h }, labelW, labelH, canvas.width, canvas.height);
            nextAnchors[i] = anchor;

            ctx.fillStyle = "rgba(0,0,0,0.65)";
            ctx.fillRect(anchor.x, anchor.y, labelW, labelH);

            ctx.fillStyle = "white";
            lines.forEach((l, j) => {
              ctx.fillText(l, anchor.x + pad, anchor.y + pad + (j + 1) * lineH - 5);
            });
          }

          smoothedFacesRef.current = nextSmoothed;
          anchorFacesRef.current = nextAnchors;
        } finally {
          busy = false;
        }
      })();
    };

    loop();
    return () => cancelAnimationFrame(rafId);
  }, [running, flipX]);

  const videoStyle = flipX ? ({ transform: "scaleX(-1)" } as React.CSSProperties) : undefined;

  return (
    <div className="app-root bottomPad" data-theme={theme}>
      <div className="container">
        <div className="header">
          <h2 className="title">On-device Facial Expression (Webcam)</h2>
          <button className="btn" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle theme">
            Theme: {theme}
          </button>
        </div>

        {mobile && <div className="warn">Mobile may be slower/less reliable. Best on desktop Chrome/Edge.</div>}

        <div className="layout">
          <div className="cameraBox">
            {running && (
              <button
                className="btn"
                onClick={() => setFlipX((v) => !v)}
                style={{
                  position: "absolute",
                  left: 10,
                  top: 10,
                  zIndex: 20,
                  padding: "8px 10px",
                  borderRadius: 10,
                }}
              >
                Flip Camera
              </button>
            )}

            <video ref={videoRef} className="videoEl" playsInline muted style={videoStyle} />
            <canvas ref={canvasRef} className="canvasEl" />
            {!running && <div className="offOverlay">Camera is off</div>}
          </div>

          <div className="sidebar">
            <div className="row">
              <button className="btn" onClick={startCamera} disabled={!ready || running}>
                Start Camera
              </button>
              <button className="btn" onClick={stopCamera} disabled={!running}>
                Stop
              </button>
            </div>

            <div className="card">
              <div className="cardTitle">Status</div>
              <div>{status}</div>
              <div className="subtext">Privacy: All processing runs on your device. No video is uploaded.</div>
              <div className="smallsubtext">Note: Display probabilities are calibrated to reduce neutral dominance.</div>
            </div>

            <div className="card">
              <div className="cardTitle">Metrics</div>
              <div>Model load: {modelLoadMs ?? "—"} ms</div>
              <div>Render FPS: {renderFps}</div>
              <div>Inference rate: {inferHz} Hz</div>
            </div>
          </div>
        </div>
      </div>

      <div className="fixedNote">Note: predicts facial expression categories (model confidence), not internal emotional state.</div>
    </div>
  );
}