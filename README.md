# FLUME — urban wind in the browser

An interactive demo that predicts the wind field inside a city, at a chosen
height, for any wind direction and speed. The prediction runs entirely in the
visitor's browser on their own GPU (WebGPU); there is no server-side
computation and no data leaves the machine.

## Running it

Requirements:

- **Desktop Chrome or Edge** (a version with WebGPU — any current release).
  Developed and tested on desktop Chrome; other browsers with WebGPU may work
  but have not been tested.
- A GPU. An integrated GPU works; a discrete one is faster.
- **Python 3** (standard library only), only to serve the files locally.

```
python3 serve.py
```

then open **http://localhost:8000/** . To use another port: `python3 serve.py 8080`.

Two things that do *not* work, and why:

- **Double-clicking `index.html`.** The page loads JavaScript modules and data
  files, which browsers only allow over http(s).
- **Opening it by the machine's IP address over plain http** (e.g.
  `http://192.168.1.5:8000`). WebGPU only runs on `https://` pages or on
  `http://localhost`. To make it reachable by other people, put this folder on
  any static web host that serves https; it needs no server-side code.

## Using it

The left panel is grouped by how long an action takes:

| Group | Controls | What happens |
|---|---|---|
| **City & height** | pick a city, upload your own, set the height | the geometry is recomputed, then the wind is predicted — the slowest action; the field fills in piece by piece, downwind |
| **Wind** | direction (drag the dial or type degrees), speed | direction re-predicts the field (quicker, no geometry step); speed is instant |
| **Display** | Speed, TKE, Temp, u, v, w; particles | instant, nothing is recomputed |

- **Speed** is the wind speed magnitude; **TKE** turbulent kinetic energy;
  **Temp** temperature; **u** the wind component along the wind direction,
  **v** across it, **w** vertical.
- **Particles**: *white, over the field* draws moving streaks on top of the
  coloured field; *coloured, no field* hides the field and colours the streaks
  by the selected quantity instead.
- **Mouse**: left-drag pans, right-drag tilts the view, the wheel zooms,
  double-click resets the view. The viewing direction is deliberately fixed,
  so the city never turns — only the wind does.
- The **scale bar** above the colour bar gives the size of the scene at the
  centre of the view; the colour bar gives the range of the displayed quantity.
- The built-in cities are 1.2 km × 1.2 km, except *Test city 302* at
  3.0 km × 3.0 km, which takes noticeably longer to compute.

## Uploading your own city

A height map as a **PNG, grayscale, any size**:

- one pixel = **4 m** on the ground;
- pixel value = **number of 1.5 m building layers** at that spot
  (0 = open ground, at most 160, i.e. 240 m);
- rows run along Y, columns along X; a 0° wind blows along +X (towards the right
  of the image).

A colour image is accepted but only its red channel is read. Values above 160
are rejected, since they are almost certainly metres or centimetres rather than
layer counts. Larger maps take proportionally longer.

## What is in this folder

```
index.html, style.css     the page
src/                      application code (plain JavaScript modules, no build step)
src/compute/              the model and its geometric input, running on WebGPU
vendor/three.module.js    three.js r169, for the 3-D view
assets/model/             the trained model: metadata.json + weights.bin
assets/cities/            the built-in cities (PNG height maps + manifest.json)
assets/colorbar.json      fixed colour scales per quantity and height
serve.py                  the local server described above
```

## The model

- A 2-D FLUME neural operator (Fourier neural operator, 4 layers, width 48,
  12 × 12 Fourier modes) with Tucker-factorised spectral weights,
  ranks (4, 22, 22, 6, 6): **221,877 parameters, 0.42 MiB**.
- Input: the building layout of the city, encoded as distance fields at the
  chosen height; output: wind components, temperature and TKE on a horizontal
  slice.
- The city is predicted in overlapping 64 × 64-cell tiles (4 m cells) aligned
  with the wind, which are blended together.
- Wind speed is applied by scaling: velocity scales with the speed, TKE with
  its square, temperature not at all. The reference speed is 6.278 m/s.

These are model predictions, not a CFD simulation; treat them as an
approximation of the flow.

## If something goes wrong

The page shows an error box if WebGPU is unavailable. Progress and diagnostic
details are written to the browser console (press **F12**, *Console* tab);
include them when reporting a problem.
