// Turning a predicted slice into what the user sees: wind-speed scaling,
// colour mapping and particle advection.
//
// Model output order is [u, w, v, T, TKE] (disk order).  u and v are the
// streamwise and lateral components in the wind frame and are displayed that
// way: u is always along the wind, v always across it, so both keep their
// meaning and their colour range as the wind direction turns.  Speed, w, T and
// TKE are rotation invariant, so every display variable is direction-agnostic.
//
// Particles are a separate, orthogonal overlay handled by particles.js; they
// are advected on the GPU in this same wind-frame grid, so nothing here needs
// city axes any more.

// `label` goes on the button, `legend` under the colour bar where there is
// room to say which direction a signed component is measured along, and `ramp`
// names the colour map in colormaps.js.  The map is chosen per variable, not
// per family: |V| and TKE are both magnitudes yet want different ramps.
import { toHalf } from "./compute/context.js";
import { rampColour } from "./colormaps.js";

// The field is drawn everywhere; nothing is masked per cell.
//
// Both edges the slice needs already exist, and both are exact to the pixel:
// buildings are lit 3-D geometry and hide the field behind them through the
// depth test, and the city outline is cut by the world-space clipping planes
// carried on the slice quad.  Re-deriving either of them from a 4 m cell grid
// and painting the result could only be coarser than what it covered up, and
// what it produced was a staircase.  The mask is a leftover from when the
// slice was a flat image with neither depth nor clipping planes.
const OPAQUE = 1;
// Variables whose scale is symmetric about zero; the rest start at zero.
const SYMMETRIC = new Set(["u", "v", "w"]);
const FIT_SAMPLES = 60000;

export const VARIABLES = [
  // Order and wording are for a reader with no background: the three physical
  // quantities first, then the three components the first of them splits into.
  { id: "speed", label: "Speed", legend: "|V| — speed",
    unit: "m/s", ramp: "turbo" },
  { id: "tke", label: "TKE", legend: "TKE — turbulence",
    unit: "m²/s²", ramp: "viridis" },
  { id: "T", label: "Temp", legend: "Temperature",
    unit: "°C", ramp: "inferno" },
  { id: "u", label: "u", legend: "u — along wind",
    unit: "m/s", ramp: "icefire" },
  { id: "v", label: "v", legend: "v — across wind",
    unit: "m/s", ramp: "icefire" },
  { id: "w", label: "w", legend: "w — vertical",
    unit: "m/s", ramp: "icefire" },
];

const CHANNEL = { u: 0, w: 1, v: 2, T: 3, tke: 4 };

/** Physical value of one display variable at a cell, after speed scaling. */
export function sample(slice, cell, variable, speedFactor) {
  return sampleAt(slice.values, cell * 5, slice.theta, variable, speedFactor);
}

/**
 * Same mapping, addressed by an explicit base offset so it also serves the
 * progressive path, whose read-back is packed as [c0..c4, solid, coverage].
 * `theta` is accepted but unused: every display variable is now either a
 * wind-frame component or a rotation invariant.  Only the particle layer,
 * which moves points through the scene, still needs the city frame.
 */
export function sampleAt(values, base, theta, variable, speedFactor) {
  const u = values[base + CHANNEL.u] * speedFactor;
  const v = values[base + CHANNEL.v] * speedFactor;
  switch (variable) {
    // Wind-frame components, shown unrotated: u is always along the wind and v
    // always across it, whatever the wind direction.  Rotating them into city
    // axes would make each button swap meaning as the direction turns, and
    // would force one fixed colour scale to cover both the along-wind range
    // and the four-times-narrower cross-wind one.
    case "u": return u;
    case "v": return v;
    case "w": return values[base + CHANNEL.w] * speedFactor;
    case "speed": {
      const w = values[base + CHANNEL.w] * speedFactor;
      return Math.sqrt(u * u + v * v + w * w);
    }
    case "T": return values[base + CHANNEL.T];
    case "tke": return values[base + CHANNEL.tke] * speedFactor * speedFactor;
    default: return 0;
  }
}

/**
 * The fixed colour scale for one variable at one height.
 *
 * Every scale has one free limit; the other end is pinned by the design, so a
 * table of one number per variable per height is enough.  Fixing them is what
 * makes colours comparable between wind directions, between heights and
 * between cities.
 *
 * The limits are quoted at the reference inflow, so they scale with the wind
 * speed the user picks: linearly for the velocity components, with the square
 * for TKE.  Temperature does not scale at all.
 */
export function scaleFor(table, variable, layer, quantile, speedFactor) {
  if (variable === "T") return table.fixed.T.slice();
  const chosen = table.quantiles[quantile]
    || table.quantiles[Object.keys(table.quantiles)[0]];
  const series = chosen[variable];
  const index = Math.min(series.length - 1, Math.max(0, Math.round(layer)));
  const limit = series[index]
    * (variable === "tke" ? speedFactor * speedFactor : speedFactor);
  return table.shape[variable] === "symmetric" ? [-limit, limit] : [0, limit];
}

// The ramps are seaborn's, baked into colormaps.js as tables.  Everything the
// viewer shows goes through here: the legend canvas draws with it directly and
// the shader samples a 256-entry texture generated from it, so there is exactly
// one definition of every colour.
function colour(ramp, t) {
  return rampColour(ramp, t);
}

/**
 * Pack the slice for the display shader: RGBA16F holding
 * [value, alpha, unused, unused] per cell.
 *
 * The value is stored at the reference wind speed and is NOT colour-mapped
 * here.  Mapping happens in the fragment shader, so changing the colour range,
 * the wind speed or the percentile costs two uniforms instead of re-colouring
 * and re-uploading every cell -- which is what made a smooth transition
 * impossible on the large city, at roughly half a microsecond per cell.
 *
 */
export function fieldTexels(slice, variable) {
  const cells = slice.solid.length;
  const raw = new Float32Array(cells * 4);
  for (let cell = 0; cell < cells; cell++) {
    const offset = cell * 4;
    raw[offset] = sample(slice, cell, variable, 1);
    raw[offset + 1] = OPAQUE;
  }
  return toHalf(raw);
}

/** 256-entry colour ramp, built from the same colour() the legend uses. */
export function paletteBytes(ramp) {
  const bytes = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = colour(ramp, i / 255);
    bytes[i * 4] = r; bytes[i * 4 + 1] = g; bytes[i * 4 + 2] = b;
    bytes[i * 4 + 3] = 255;
  }
  return bytes;
}

/**
 * The scale that actually fits this slice, at the reference wind speed.
 *
 * Sampled on a stride rather than sorted whole: a percentile off ~60k cells is
 * stable well past the third digit, and sorting 60k values takes a few
 * milliseconds where sorting the large city's 1.1M would take a few hundred --
 * a hitch landing exactly when the fill ends.  Returns null when the variable
 * has a pinned scale or the slice carries nothing to fit.
 */
export function fitRange(slice, variable, quantile) {
  if (variable === "T") return null;
  const id = variable;
  const cells = slice.solid.length;
  const step = Math.max(1, Math.floor(cells / FIT_SAMPLES));
  const symmetric = SYMMETRIC.has(id);
  const values = [];
  for (let cell = 0; cell < cells; cell += step) {
    if (slice.solid[cell] !== 0) continue;
    const value = sample(slice, cell, id, 1);
    values.push(symmetric ? Math.abs(value) : value);
  }
  if (values.length < 32) return null;
  values.sort((a, b) => a - b);
  const index = Math.min(values.length - 1,
    Math.floor(values.length * (Number(quantile) / 100)));
  const limit = values[index];
  if (!(limit > 0)) return null;
  return symmetric ? [-limit, limit] : [0, limit];
}

/**
 * Pack one progressive rectangle in the same layout.
 *
 * Alpha carries the blend coverage, so the advancing edge of the fill fades in
 * with the Hann window instead of arriving as a hard 64x64 square: a cell that
 * only the tail of one patch has reached is nearly transparent and firms up as
 * its neighbours land.  Cells no patch has reached yet stay fully transparent.
 */
export function regionTexels(update, variable) {
  const { values, stride, channels, theta, width, height } = update;
  const raw = new Float32Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const base = i * stride;
    const offset = i * 4;
    raw[offset] = sampleAt(values, base, theta, variable, 1);
    // Coverage is the only thing gating a cell now: it is what makes the fill
    // fade in, and it is zero exactly where no patch has reached yet.
    raw[offset + 1] = OPAQUE * Math.min(1, values[base + channels + 1]);
  }
  return toHalf(raw);
}

/** Particle state advected by the city-frame horizontal velocity. */

/**
 * The flow the particles are advected by, as plain arrays over the wind-frame
 * grid.  u and v go in unrotated: the grid IS the wind frame, so u already
 * points along +X and v along +Y, and the plane carries the rotation into the
 * scene.  Values are at the reference inflow; the caller scales the time step.
 */
/**
 * An empty flow, for a prediction that has not started arriving yet.
 *
 * Every cell begins blocked, so a particle can neither sit in one nor spawn
 * there.  Regions are merged in as the fill reaches them, which is what lets
 * the streaks grow with the prediction instead of waiting for all of it.
 */
export function emptyFlowField(width, height) {
  const cells = width * height;
  return {
    u: new Float32Array(cells), v: new Float32Array(cells),
    value: new Float32Array(cells), solid: new Uint8Array(cells).fill(1),
    width, height,
  };
}

/** Merge one finished rectangle of a progressive fill into a flow. */
export function mergeFlowRegion(field, update, variable) {
  const { values, stride, channels, theta, width, height, x0, y0 } = update;
  for (let i = 0; i < width * height; i++) {
    const base = i * stride;
    // Half-blended cells would advect particles along a velocity that is still
    // changing; wait until the overlap-add for that cell is complete.
    if (values[base + channels + 1] < 0.999) continue;
    const cell = (y0 + Math.floor(i / width)) * field.width + (x0 + i % width);
    field.value[cell] = sampleAt(values, base, theta, variable, 1);
    if (values[base + channels] !== 0) continue;   // building or outside
    field.u[cell] = values[base + CHANNEL.u];
    field.v[cell] = values[base + CHANNEL.v];
    field.solid[cell] = 0;
  }
}

export function flowField(slice, variable) {
  const cells = slice.solid.length;
  const u = new Float32Array(cells);
  const v = new Float32Array(cells);
  // The displayed quantity travels with the flow so a particle can be tinted
  // by it.  Stored at the reference inflow, like everything else: changing the
  // wind speed scales the colour range by the same factor, so the normalized
  // position on the ramp is unchanged and this never has to be rebuilt.
  const value = new Float32Array(cells);
  for (let cell = 0; cell < cells; cell++) {
    value[cell] = sample(slice, cell, variable, 1);
    if (slice.solid[cell] !== 0) continue;
    u[cell] = slice.values[cell * 5 + CHANNEL.u];
    v[cell] = slice.values[cell * 5 + CHANNEL.v];
  }
  return { u, v, value, solid: slice.solid,
           width: slice.width, height: slice.height };
}

// colour() is the single definition of every colour in the page: the legend
// canvas calls it directly and the shader's 256-entry ramp is generated from
// it, so the two can never drift apart.
export { colour };
