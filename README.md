<img src="public/logo.png" alt="ARC Whitebox Estimation Challenge logo" style="height: 80px;">

# WhestBench Explorer

Interactive React app for visualizing small random MLPs and estimator behavior.

[![Deploy to GitHub Pages](https://github.com/AIcrowd/whestbench-explorer/actions/workflows/deploy.yml/badge.svg)](https://github.com/AIcrowd/whestbench-explorer/actions/workflows/deploy.yml)

**Live demo:** https://aicrowd.github.io/whestbench-explorer/

WhestBench Explorer is the educational and debugging companion to [whestbench](https://github.com/AIcrowd/whestbench-public) (the library + CLI). It is **not** the submission interface — official local scoring lives in `whest run --estimator <path> --runner server` in the main repo.

All computation runs in your browser (Web Workers + TensorFlow.js); there is no backend.

## Quick start

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## What you should see

| Panel | Description |
|---|---|
| Network Graph | Layer/neuron structure with value-oriented coloring |
| Signal Heatmap | Layer-wise activation behavior |
| Estimator Comparison | Per-layer error across available estimators |
| Controls | Width, depth, seed, and budget knobs |

## Suggested use during estimator iteration

1. Reproduce a pattern on small MLPs.
2. Inspect where errors spike by depth.
3. Convert that intuition into estimator logic.
4. Re-test with the official local scorer in [whestbench](https://github.com/AIcrowd/whestbench-public).

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Start the Vite dev server |
| `npm run build` | Build a static bundle into `dist/` |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run ESLint |
| `npm run test` | Run the Vitest suite |

## Deployment

Pushes to `main` trigger [.github/workflows/deploy.yml](.github/workflows/deploy.yml), which builds with Vite and deploys `dist/` to GitHub Pages at https://aicrowd.github.io/whestbench-explorer/.

The Vite `base` is set to `/whestbench-explorer/` so the bundle resolves correctly under the project Pages path.
