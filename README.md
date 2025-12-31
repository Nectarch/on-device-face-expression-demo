# On-device Facial Expression Demo (Webcam)

A client-side web app that runs facial expression classification in real time from your laptop camera and renders:

- A face bounding box (single or multiple faces supported by the detector)
- Top expression probabilities (model confidence)
- Basic performance metrics (FPS / inference rate)

**Privacy:** All processing runs locally in your browser. No video is uploaded.

**Technical report (DOI):** https://doi.org/10.5281/zenodo.18109914

## Live Demo

- https://on-device-face-expression-demo-by-michael.vercel.app

## Repository

- https://github.com/Nectarch/on-device-face-expression-demo

## Features

- Real-time webcam pipeline (Start/Stop)
- Face bounding box + top-3 expression probabilities
- On-device inference (face-api.js)
- Theme toggle (light/dark)
- Mirror option (Flip Camera)
- Hidden Easter egg

## Tech Stack

- React + TypeScript + Vite
- face-api.js (Tiny Face Detector + Expression model)
- All model files served from `/public/models`

## How it works

1. Loads face-api.js models from `/public/models`
2. Captures webcam frames via `getUserMedia`
3. Runs face detection + expression inference at a throttled rate
4. Smooths displayed probabilities with an exponential moving average (EMA)
5. Renders overlays on a canvas aligned with the video

## Run locally

Requirements: Node.js 18+ recommended.

```bash
npm install
npm run dev
```

## Build
```bash
npm run build
npm run preview
```
## Notes and Limitations
- Predictions are expression categories from the model, not a measurement of internal emotional state.

- Performance varies with lighting, camera quality, and pose.

## Citation

Michael Farrell Gunawan (2025). *Privacy-Preserving Real-Time Facial Expression Classification in the Browser: System Design and Tradeoff Evaluation (Technical Report).* Zenodo. https://doi.org/10.5281/zenodo.18109914
