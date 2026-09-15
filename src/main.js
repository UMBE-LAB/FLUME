// Application wiring: state, the recompute policy, and display updates.
//
// Recompute policy (WEB_DEMO_HANDOFF section 5.3):
//   city or upload -> MDDF at the current height, then predict
//   height         -> MDDF at the new height, then predict
//   direction      -> predict only; MDDF rays are never re-traced
//   speed          -> post-process the existing prediction
//   variable       -> re-render the existing prediction
// Only the latest requested state is honoured; stale work is discarded.

import * as THREE from "../vendor/three.module.js";
import { Context } from "./compute/context.js";
import { Predictor } from "./compute/predict.js";
import { loadManifest, loadBuiltIn, loadUpload,
         HORIZONTAL_SPACING_M, VERTICAL_SPACING_M } from "./city.js";
import { Viewer } from "./viewer.js";
import { buildControls } from "./controls.js";
import { VARIABLES, colour, fieldTexels, regionTexels, paletteBytes,
         scaleFor, fitRange, flowField, emptyFlowField,
         mergeFlowRegion } from "./field.js";
import { ParticleFlow } from "./particles.js";

// One display refresh per screen refresh at most.
const FRAME_MS = 1000 / 60;

// The percentile the colour scales are quoted at.  It was a menu while the
// three candidates were being compared on real cities; 99.9 won, and the menu
// went with it: a control the user is never meant to touch again is only a
// question they have to answer every time they read the panel.
const QUANTILE = "99.9";

const state = {
  city: null, layer: 26, theta: 0, speed: 6.278, referenceSpeed: 6.278,
  variable: "speed", slice: null, pending: false, queued: false,
  particles: "white",          // streaks are on when the page opens
};

// Progress and diagnostics go to the browser console (F12), not the page: a
// visitor never needs them, and whoever supports the page still has them.
const startedAt = performance.now();
function log(message, bad = false) {
  const line =
    `[FLUME ${((performance.now() - startedAt) / 1000).toFixed(2)}s] ${message}`;
  if (bad) console.warn(line); else console.info(line);
}

let controls;
let viewer;
let predictor;
let particles;      // ParticleFlow, orthogonal to the field variable
let sceneTexture = null;      // RGBA16F [value, alpha, unused, unused]
let paletteTexture = null;
let paletteKind = null;
let colorbar = null;          // fixed scales, one limit per variable per height
let lastDrawAt = 0;
// The colour range in force, in reference-speed units.  Wind speed is applied
// on the way to the shader, so changing it never touches the texture.
let referenceRange = [0, 1];
let glide = null;

const GLIDE_MS = 420;

/** The fixed table scale, in reference-speed units. */
function tableScale(variable) {
  return scaleFor(colorbar, variable, state.layer, QUANTILE, 1);
}

/** Wind speed enters here and nowhere else: linear, squared for TKE, none for T. */
function speedScaled(range, variable) {
  if (variable === "T") return range.slice();
  const factor = state.speed / state.referenceSpeed;
  const power = variable === "tke" ? factor * factor : factor;
  return [range[0] * power, range[1] * power];
}

/** Push a reference-speed range to the shader and the legend. */
function applyRange(range) {
  referenceRange = range;
  const shown = speedScaled(range, state.variable);
  viewer.setRange(shown[0], shown[1]);
  controls.setLegendRange(shown[0], shown[1]);
  // Tinted particles read the same scale as the slice, so they follow it
  // through the fitted-range glide and the percentile switch as well.
  if (state.particles === "tinted") {
    particles.setTinting(paletteBytes(rampFor(state.variable)), shown);
  }
}

/** Ease from the range in force to a new one; used after a prediction lands. */
function glideRange(target) {
  if (glide) cancelAnimationFrame(glide);
  const from = referenceRange.slice();
  const started = performance.now();
  const step = () => {
    const t = Math.min(1, (performance.now() - started) / GLIDE_MS);
    const e = 0.5 - 0.5 * Math.cos(Math.PI * t);   // ease in and out
    applyRange([from[0] + (target[0] - from[0]) * e,
                from[1] + (target[1] - from[1]) * e]);
    glide = t < 1 ? requestAnimationFrame(step) : null;
  };
  glide = requestAnimationFrame(step);
}

/** Refit the scale to the slice actually predicted, then ease over to it. */
function refit({ animate = true } = {}) {
  if (!state.slice) return;
  const fitted = fitRange(state.slice, state.variable, QUANTILE)
              || tableScale(state.variable);
  if (animate) glideRange(fitted);
  else { if (glide) cancelAnimationFrame(glide); glide = null; applyRange(fitted); }
}

/** Upload the whole field for the current variable and match the palette. */
function uploadField() {
  if (!state.slice || !sceneTexture) return;
  const definition = VARIABLES.find((entry) => entry.id === state.variable);
  sceneTexture.image.data.set(fieldTexels(state.slice, state.variable));
  sceneTexture.needsUpdate = true;
  usePalette(definition.ramp);
  controls.setLegend(definition);
}

/** The colour map a variable is drawn with. */
function rampFor(variable) {
  return VARIABLES.find((entry) => entry.id === variable).ramp;
}

function usePalette(ramp) {
  if (paletteKind === ramp) return;
  if (paletteTexture) paletteTexture.dispose();
  paletteTexture = new THREE.DataTexture(
    paletteBytes(ramp), 256, 1, THREE.RGBAFormat);
  // The ramp holds sRGB display values.  Without this the GPU hands them to the
  // shader as if they were linear and the output encode lifts them a second
  // time: near-black 31 comes out at 98, and the whole slice goes milky.
  paletteTexture.colorSpace = THREE.SRGBColorSpace;
  paletteTexture.magFilter = THREE.LinearFilter;
  paletteTexture.minFilter = THREE.LinearFilter;
  paletteTexture.needsUpdate = true;
  paletteKind = ramp;
  viewer.setPalette(paletteTexture);
}

function describeCity(city) {
  const [spanY, spanX] = city.extent;
  // One line, shown under the picker, so it says what a reader can act on:
  // how big the place is and how tall it gets.  The cell count is in the log.
  return `${(spanX / 1000).toFixed(1)} × ${(spanY / 1000).toFixed(1)} km · ` +
         `tallest ${city.maximumHeight.toFixed(0)} m` +
         (city.notes.length ? ` · ${city.notes.join("; ")}` : "");
}

/** Coalescing scheduler: never queue more than one obsolete request. */
function schedule(work) {
  if (state.pending) { state.queued = work; return; }
  state.pending = true;
  controls.setBusy(true);
  (async () => {
    try {
      await work();
    } catch (error) {
      console.error(error);
      controls.setStatus(`error: ${error.message}`);
      log(`FAILED: ${error.message}`, true);
      window.__flumeError?.(`${error.message}\n${error.stack || ""}`);
    } finally {
      state.pending = false;
      const next = state.queued;
      state.queued = null;
      controls.setBusy(false);
      if (next) schedule(next);
    }
  })();
}

async function recompute({ height = false } = {}) {
  if (!state.city) return;
  const started = performance.now();
  let traced = 0;
  if (height) {
    const t0 = performance.now();
    await predictor.setHeight(state.layer);
    traced = performance.now() - t0;
    log(`MDDF layer ${state.layer}: ${(traced / 1000).toFixed(2)}s`);
  }
  // Fill the slice in as the patches land.  requestAnimationFrame's cadence is
  // the pacing signal: refreshing faster than the display refreshes produces
  // frames nobody sees, and refreshing slower only adds lag, since the extra
  // work does not grow with the refresh rate.
  const gaps = [];
  let refreshes = 0;
  let previousAt = 0;
  const slice = await predictor.predict(state.theta, {
    onStart: beginFill,
    shouldDraw: () => performance.now() - lastDrawAt >= FRAME_MS,
    onDraw: (update) => {
      paintFill(update);
      const now = performance.now();
      if (previousAt) gaps.push(now - previousAt);
      previousAt = now;
      lastDrawAt = now;
      refreshes += 1;
      controls.setStatus(
        `predicting… ${Math.round(100 * update.readyCount / update.patchCount)}%`);
    },
  });
  if (refreshes) {
    const mean = gaps.length
      ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;
    log(`progressive fill: ${refreshes} refreshes, ` +
        `${mean.toFixed(0)} ms apart, ` +
        `${(slice.patchCount / refreshes).toFixed(1)} patches each`);
  }
  state.slice = slice;
  // The prediction that just landed is the truth; put all of it on the GPU
  // rather than trusting the sum of the rectangles uploaded along the way.
  // The closing full-grid blend makes this byte-identical to a fill that went
  // perfectly, so there is nothing to see -- but when a rectangle has gone
  // astray, this is what puts the field on screen instead of leaving it blank
  // until the user happens to change a variable.
  uploadField();
  log(`predict ${slice.patchCount} patches -> ${slice.width}x${slice.height}` +
      `, ${(slice.milliseconds / 1000).toFixed(2)}s` +
      ` · uncovered ${slice.uncovered} · non-finite ${slice.nonFinite}`,
      slice.uncovered > 0 || slice.nonFinite > 0);
  // Swap the merged-in-pieces flow for the exact final one, without clearing
  // the streaks that have been growing all along.
  if (state.particles !== "off") {
    particles.replaceField(flowField(state.slice, state.variable));
  }
  applyStage();
  refit({ animate: true });
  log("slice rendered · colour scale easing to the fitted range");
  const total = performance.now() - started;
  // Plain language only.  The GPU timings and the patch count are engineering
  // detail and were logged a few lines above, which is where they belong: this
  // box is two lines tall and a first-time reader gets nothing from them.
  controls.setStatus(
    `${state.city.label} at ` +
    `${((state.layer + 0.5) * VERTICAL_SPACING_M).toFixed(0)} m` +
    ` · ready in ${(total / 1000).toFixed(1)} s`);
}

/**
 * Start a progressive fill: put an empty texture of the right shape on the
 * slice plane so the city stays visible and the wind field arrives into it.
 */
function beginFill(info) {
  const layout = {
    width: info.gridW, height: info.gridH,
    originX: info.originX, originY: info.originY, theta: info.theta,
  };
  if (sceneTexture) sceneTexture.dispose();
  sceneTexture = new THREE.DataTexture(
    new Uint16Array(info.gridW * info.gridH * 4),
    info.gridW, info.gridH, THREE.RGBAFormat, THREE.HalfFloatType);
  sceneTexture.magFilter = THREE.LinearFilter;
  sceneTexture.minFilter = THREE.LinearFilter;
  sceneTexture.needsUpdate = true;
  const definition = VARIABLES.find((entry) => entry.id === state.variable);
  usePalette(definition.ramp);
  controls.setLegend(definition);
  // Start on the fixed table scale: nothing has been predicted yet, so there
  // is nothing to fit to.  The fit happens once the field is complete.
  if (glide) cancelAnimationFrame(glide);
  glide = null;
  applyRange(tableScale(state.variable));
  // setSlice renders, which is what actually uploads the texture; the partial
  // uploads that follow need it resident.
  viewer.setSlice(sceneTexture, layout,
                  (state.layer + 0.5) * VERTICAL_SPACING_M);
  viewer.setInflow(info.theta);
  // A fresh, empty flow on the new grid: the old one belongs to the previous
  // wind direction and its grid no longer matches this plane.
  particles.setPlane(info.gridW * HORIZONTAL_SPACING_M,
                     info.gridH * HORIZONTAL_SPACING_M);
  particles.setField(emptyFlowField(info.gridW, info.gridH),
                     HORIZONTAL_SPACING_M);
  viewer.setStreaks(particles.mesh);
  // setSlice builds a fresh mesh, which is visible by default; re-apply the
  // mode at once or the hidden field reappears for the length of the fill.
  applyStage();
}

/** Paint one rectangle of the fill and upload just that rectangle. */
function paintFill(update) {
  viewer.uploadRegion(regionTexels(update, state.variable), update.width,
                      update.height, update.x0, update.y0, sceneTexture);
  // The same read-back already carries u, v and the displayed quantity, so the
  // flow costs nothing extra to keep up to date.
  if (state.particles !== "off" && particles.field) {
    mergeFlowRegion(particles.field, update, state.variable);
    particles.setProgress(update.readyCount / update.patchCount,
                          update.x0 + update.width);
  }
}

/**
 * Everything that changes when the variable changes: the packed field, the
 * colour ramp, the legend, the fitted scale and the particle layer.  A colour
 * range change alone does none of this -- that is applyRange, two uniforms.
 */
function render() {
  if (!state.slice) return;
  uploadField();
  refit({ animate: false });
  // The tint comes from the displayed variable, so the flow carries a fresh
  // copy of it whenever that changes.
  if (state.particles !== "off") {
    installFlow();
    applyStage();
  }
}

/**
 * Point the flow at the prediction and start or stop the animation.  Nothing
 * here depends on which field is on screen: the particles are an overlay.
 */
/**
 * Three states, not two: off; white streaks over the coloured field; or
 * coloured streaks with the field hidden, so the particles alone carry it.
 */

/**
 * The single place that decides what is on screen.
 *
 * Two things vary: which particle mode is chosen, and whether a prediction is
 * in flight.  Every rule below reads both, and nothing else is allowed to
 * touch visibility -- scattering these decisions across beginFill, the mode
 * handler and the render path is what let the field flash into view mid-fill
 * in the mode whose whole point is that the field is not shown.
 *
 *   slice   visible unless the particles are carrying the field themselves
 *   streaks visible only once a prediction has landed: while one is in flight
 *           the flow field belongs to the previous wind direction, on a grid
 *           that no longer matches the plane
 *   animation runs exactly when the streaks are visible
 */
function applyStage() {
  const tinted = state.particles === "tinted";
  // The streaks run as soon as there is a flow to run on, even a partly filled
  // one: regions are merged in as the prediction reaches them, so the lines
  // grow downwind with it instead of the view sitting empty until it finishes.
  const streaks = state.particles !== "off"
    && Boolean(particles.field) && Boolean(viewer.slice);

  particles.setTinting(
    tinted ? paletteBytes(rampFor(state.variable)) : null,
    tinted ? speedScaled(referenceRange, state.variable) : null);
  viewer.showSlice(!tinted);
  viewer.showStreaks(streaks);
  if (!streaks) { viewer.stopAnimation(); viewer.render(); return; }

  // Real elapsed time, so the streak length is the same however the frame rate
  // moves, and the wind speed setting scales how fast the flow is played.
  let previous = performance.now();
  viewer.animate(() => {
    const now = performance.now();
    const delta = (now - previous) / 1000;
    previous = now;
    particles.step(delta * (state.speed / state.referenceSpeed),
                   viewer.camera, viewer.slice);
  });
}

/** Point the flow at the prediction that just landed and wipe the old streaks. */
function installFlow() {
  if (!state.slice) return;
  particles.setPlane(state.slice.width * HORIZONTAL_SPACING_M,
                     state.slice.height * HORIZONTAL_SPACING_M);
  particles.setField(flowField(state.slice, state.variable),
                     HORIZONTAL_SPACING_M);
  viewer.setStreaks(particles.mesh);
}

async function useCity(city) {
  state.city = city;
  let built = 0;
  for (let i = 0; i < city.layers.length; i++) if (city.layers[i] > 0) built++;
  log(`city ${city.label}: ${city.nx}x${city.ny}, ${built} built cells, ` +
      `tallest ${city.maximumLayers} layers`);
  controls.setCityInfo(describeCity(city));
  const maximumLayer = Math.max(0, Math.min(159, city.maximumLayers + 20));
  controls.setHeightBounds(maximumLayer);
  state.layer = controls.setHeight(Math.min(state.layer, maximumLayer));

  // Geometry first: the scene must be verifiable without the compute chain.
  viewer.setCity(city);
  log(`buildings drawn: ${viewer.buildingCount} instances`);
  log(viewer.diagnostics());

  const grid = predictor.setCity(city.layers, city.ny, city.nx);
  log(`MDDF grid ${grid.gx}x${grid.gy} (margin ${grid.margin})`);
  await recompute({ height: true });
}

async function start() {
  viewer = new Viewer(document.getElementById("scene"));
  log(`viewer created · ${viewer.diagnostics()}`);
  particles = new ParticleFlow(viewer.renderer);
  // Start in the mode `state` names, exactly as choosing it from the menu would.
  particles.enabled = state.particles !== "off";
  particles.setMode(state.particles);
  // The ribbon width is in pixels, so it has to follow the drawing buffer.
  viewer.onResize = (width, height) => particles.setResolution(width, height);
  viewer.resize();
  controls = buildControls({
    colour,
    onSpeed(value) { state.speed = value; applyRange(referenceRange); },
    onParticles(mode) {
      state.particles = mode;
      particles.enabled = mode !== "off";
      particles.setMode(mode);
      if (mode !== "off" && state.slice) installFlow();
      applyStage();
    },
    onVariable(id) { state.variable = id; render(); },
    onDirection(theta) {
      state.theta = theta;
      schedule(() => recompute());
    },
    onHeight(layer) {
      // Releasing the slider where it started still fires a change event, and
      // this is the most expensive action on the page; do not pay for it twice.
      if (layer === state.layer) return;
      state.layer = layer;
      schedule(() => recompute({ height: true }));
    },
    onCity(id) {
      schedule(async () => {
        controls.setStatus("loading city…");
        await useCity(await loadBuiltIn(manifest, id));
      });
    },
    onUpload(file) {
      schedule(async () => {
        controls.setStatus(`reading ${file.name}…`);
        const city = await loadUpload(file);
        controls.showUploaded(file);
        await useCity(city);
      });
    },
  });
  controls.selectVariable(state.variable);
  // Set from state rather than trusting the markup: on a reload the browser
  // may restore whichever option was last picked, and the menu would then
  // disagree with what is actually drawn.
  controls.selectParticles(state.particles);
  // The scale bar is the only thing that tells a 1.2 km city from a 3.0 km one
  // apart once the camera has refitted to each of them.  Wired after the
  // controls exist: viewer.resize() runs before buildControls and would fire
  // this callback into an undefined `controls`.
  viewer.onCamera = () => controls?.setScale(viewer.metresPerPixel());
  viewer.onCamera();

  let manifest;
  try {
    controls.setStatus("starting the GPU…");
    log("requesting WebGPU adapter");
    const context = await Context.create();
    log(`GPU: ${await context.describe()} · max storage buffer `
      + `${(context.device.limits.maxStorageBufferBindingSize / 2 ** 20).toFixed(0)} MiB`);
    const seen = new Set();
    context.device.addEventListener("uncapturederror", (event) => {
      const first = event.error.message.split("\n")[0];
      if (seen.has(first)) return;
      seen.add(first);
      window.__flumeError?.(`WebGPU: ${first}`);
    });
    controls.setStatus("loading the model…");

    const [metadata, blob, scales] = await Promise.all([
      fetch("assets/model/metadata.json").then((r) => r.json()),
      fetch("assets/model/weights.bin").then((r) => r.arrayBuffer()),
      fetch("assets/colorbar.json").then((r) => r.json()),
    ]);
    colorbar = scales;
    // scaleFor falls back to the first percentile in the table when this one is
    // missing, so a mismatch degrades instead of throwing; say so in the log
    // rather than letting it pass unnoticed.
    const offered = Object.keys(colorbar.quantiles);
    log(`colour scales: ${colorbar.layers} heights at the ${QUANTILE}th ` +
        `percentile (table carries ${offered.join("/")})`,
        !offered.includes(QUANTILE));
    log(`model ${metadata.label}: ${(blob.byteLength / 2 ** 20).toFixed(2)} MiB, ${metadata.model.input_channels}->${metadata.model.output_channels} channels`);
    predictor = new Predictor(context, metadata, blob, { batchSize: 8 });
    log(`operator ready · ${context.useF16 ? "f16" : "f32"} storage · `
      + `${(context.peakBytes / 2 ** 20).toFixed(1)} MiB allocated`);
    state.referenceSpeed = metadata.wind.reference_speed_m_s;
    state.speed = state.referenceSpeed;
    controls.speed.silent(state.referenceSpeed);
    manifest = await loadManifest();
    log(`${manifest.cities.length} built-in cities`);
    controls.setCities(manifest, manifest.cities[0].id);
    log(`${await context.describe()} · model ${metadata.label}, ` +
        `${(blob.byteLength / 2 ** 20).toFixed(2)} MiB, validation RMSE ` +
        `${metadata.validation_macro_rmse.toFixed(4)} · ` +
        `${context.useF16 ? "f16" : "f32"} storage`);

    schedule(async () => {
      await useCity(await loadBuiltIn(manifest, manifest.cities[0].id));
    });
  } catch (error) {
    console.error(error);
    log(`FAILED: ${error.message}`, true);
    controls.showError(
      `${error.message}\n\nThis demo needs WebGPU: desktop Chrome or Edge, ` +
      `served over https:// or http://localhost.`);
    controls.setStatus("unavailable");
  }
}

start();
