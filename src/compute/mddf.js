// Four-signal MDDF on the expanded grid, plus the fused "assemble" pass that
// turns it into model input patches for an arbitrary wind direction.
//
// The ray tracing and Fourier kernels are ports of prepare_mddf.py, verified
// against its CPU reference (probe error 0 m).  They run once per query
// height.  A wind-direction change never re-traces rays: shifting the angular
// reference by theta is a phase rotation of the stored coefficients,
//     G_m = C_m * exp(i*m*theta),
// applied inside the assemble pass while patches are sampled obliquely.

export const ANGLE_COUNT = 180;
export const SIGNAL_COUNT = 4;
export const CHANNELS_PER_SIGNAL = 7;   // [DC, Re1..Re3, Im1..Im3]
export const FIELD_CHANNELS = SIGNAL_COUNT * CHANNELS_PER_SIGNAL;
export const VERTICAL_SPACING_M = 1.5;
export const HORIZONTAL_SPACING_M = 4.0;
const DISTANCE_CUTOFF_M = 1600.0;
// WGSL infers an integer from a literal without a decimal point,
// so every float constant is templated through this.
const wgslFloat = (value) => value.toFixed(6);
// prepare_mddf.py used 640; keeping it identical preserves the training-time
// definition, including where very long diagonal rays stop early.
const RAY_STEP_LIMIT = 640;

const TRACE = `
struct P { gy: u32, gx: u32, chunk: u32, pixelOffset: u32,
           pixelCount: u32, pad0: u32, pad1: u32, pad2: u32,
           queryHeight: f32, pad3: f32, pad4: f32, pad5: f32 };
@group(0) @binding(0) var<storage, read> height: array<f32>;
@group(0) @binding(1) var<storage, read> cosT: array<f32>;
@group(0) @binding(2) var<storage, read> sinT: array<f32>;
@group(0) @binding(3) var<storage, read_write> raw: array<f32>;
@group(0) @binding(4) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  let angles = ${ANGLE_COUNT}u;
  if (g >= u.pixelCount * angles) { return; }
  let localPixel = g / angles;
  let angle = g - localPixel * angles;
  let pixel = u.pixelOffset + localPixel;
  let nx = i32(u.gx);
  let ny = i32(u.gy);
  var x = i32(pixel % u.gx);
  var y = i32(pixel / u.gx);
  let gxr = cosT[angle] / ${wgslFloat(HORIZONTAL_SPACING_M)};
  let gyr = sinT[angle] / ${wgslFloat(HORIZONTAL_SPACING_M)};

  var stepX = 0; var nextX = 1.0e20; var deltaX = 1.0e20;
  if (abs(gxr) >= 1.0e-10) {
    if (gxr > 0.0) { stepX = 1; nextX = 0.5 / gxr; deltaX = 1.0 / gxr; }
    else { stepX = -1; nextX = -0.5 / gxr; deltaX = -1.0 / gxr; }
  }
  var stepY = 0; var nextY = 1.0e20; var deltaY = 1.0e20;
  if (abs(gyr) >= 1.0e-10) {
    if (gyr > 0.0) { stepY = 1; nextY = 0.5 / gyr; deltaY = 1.0 / gyr; }
    else { stepY = -1; nextY = -0.5 / gyr; deltaY = -1.0 / gyr; }
  }

  let cutoff = ${wgslFloat(DISTANCE_CUTOFF_M)};
  var upperDistance = cutoff;
  var upperHeight = 0.0;
  var lowerDistance = cutoff;
  var lowerHeight = -u.queryHeight;
  var bestSlope = -1.0e20;
  var entered = 0.0;

  for (var step = 0u; step < ${RAY_STEP_LIMIT}u; step++) {
    if (x < 0 || x >= nx || y < 0 || y >= ny || entered >= cutoff) { break; }
    let leaving = min(min(nextX, nextY), cutoff);
    let roof = height[u32(y) * u.gx + u32(x)];
    if (roof >= u.queryHeight) {
      upperDistance = entered;
      upperHeight = roof - u.queryHeight;
      lowerDistance = entered;
      lowerHeight = 0.0;
      break;
    }
    if (roof > 0.0) {
      let slope = (roof - u.queryHeight) / max(leaving, 1.0e-8);
      if (slope > bestSlope) {
        bestSlope = slope;
        lowerDistance = leaving;
        lowerHeight = roof - u.queryHeight;
      }
    }
    let crossX = nextX <= nextY + 1.0e-7;
    let crossY = nextY <= nextX + 1.0e-7;
    if (crossX) { x += stepX; nextX += deltaX; }
    if (crossY) { y += stepY; nextY += deltaY; }
    entered = leaving;
  }
  let base = localPixel * angles + angle;
  let stride = u.chunk * angles;
  raw[base] = upperDistance;
  raw[stride + base] = upperHeight;
  raw[2u * stride + base] = lowerDistance;
  raw[3u * stride + base] = lowerHeight;
}`;

const FOURIER = `
struct P { gy: u32, gx: u32, chunk: u32, pixelOffset: u32,
           pixelCount: u32, pad0: u32, pad1: u32, pad2: u32,
           queryHeight: f32, pad3: f32, pad4: f32, pad5: f32 };
@group(0) @binding(0) var<storage, read> raw: array<f32>;
@group(0) @binding(1) var<storage, read_write> field: array<f32>;
@group(0) @binding(2) var<uniform> u: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 64u;
  if (g >= u.pixelCount * 4u) { return; }
  let sig = g % 4u;
  let localPixel = g / 4u;
  let angles = ${ANGLE_COUNT}u;
  var dc = 0.0;
  var re = vec3<f32>(0.0);
  var im = vec3<f32>(0.0);
  let base = sig * u.chunk * angles + localPixel * angles;
  let n = f32(angles);
  for (var a = 0u; a < angles; a++) {
    let v = raw[base + a];
    let t = 6.283185307179586 * f32(a) / n;
    dc += v;
    re += v * vec3<f32>(cos(t), cos(2.0 * t), cos(3.0 * t));
    im -= v * vec3<f32>(sin(t), sin(2.0 * t), sin(3.0 * t));
  }
  let out = (u.pixelOffset + localPixel) * ${FIELD_CHANNELS}u
          + sig * ${CHANNELS_PER_SIGNAL}u;
  field[out] = dc / n;
  field[out + 1u] = re.x / n; field[out + 2u] = re.y / n; field[out + 3u] = re.z / n;
  field[out + 4u] = im.x / n; field[out + 5u] = im.y / n; field[out + 6u] = im.z / n;
}`;

// One thread per element of the model input tensor [B, C, patch, patch].
// Fuses oblique sampling, phase rotation, normalization, solid mask and coords.
// The output element type must match what the operator reads, so the shader is
// templated on the storage type rather than assuming f32.
const assembleShader = (store) => `${store === "f16" ? "enable f16;\n" : ""}
struct P {
  batch: u32, channels: u32, mddfChannels: u32, patchSize: u32,
  gy: u32, gx: u32, margin: u32, pad0: u32,
  solidChannel: i32, coordChannel: i32, pad1: i32, pad2: i32,
  cosT: f32, sinT: f32, theta: f32, queryLayer: f32,
  centreY: f32, centreX: f32, coordYScale: f32, coordZ: f32,
  coordXCentre: f32, coordXScale: f32, pad3: f32, pad4: f32,
};
@group(0) @binding(0) var<storage, read> field: array<f32>;
@group(0) @binding(1) var<storage, read> height: array<f32>;
@group(0) @binding(2) var<storage, read> origins: array<f32>;
@group(0) @binding(3) var<storage, read> table: array<u32>;
@group(0) @binding(4) var<storage, read> norm: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<${store}>;
@group(0) @binding(6) var<uniform> u: P;

fn sampleField(ey: f32, ex: f32, channel: u32) -> f32 {
  let y0 = floor(ey); let x0 = floor(ex);
  let fy = ey - y0; let fx = ex - x0;
  let iy0 = clamp(i32(y0), 0, i32(u.gy) - 1);
  let ix0 = clamp(i32(x0), 0, i32(u.gx) - 1);
  let iy1 = clamp(iy0 + 1, 0, i32(u.gy) - 1);
  let ix1 = clamp(ix0 + 1, 0, i32(u.gx) - 1);
  let c = ${FIELD_CHANNELS}u;
  let v00 = field[(u32(iy0) * u.gx + u32(ix0)) * c + channel];
  let v01 = field[(u32(iy0) * u.gx + u32(ix1)) * c + channel];
  let v10 = field[(u32(iy1) * u.gx + u32(ix0)) * c + channel];
  let v11 = field[(u32(iy1) * u.gx + u32(ix1)) * c + channel];
  return mix(mix(v00, v01, fx), mix(v10, v11, fx), fy);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  let area = u.patchSize * u.patchSize;
  if (g >= u.batch * u.channels * area) { return; }
  let p = g % area;
  let channel = (g / area) % u.channels;
  let b = g / (area * u.channels);

  // Wind-frame position of this pixel, then the city cell it samples.
  let sx = origins[b * 2u] + f32(p % u.patchSize);
  let sy = origins[b * 2u + 1u] + f32(p / u.patchSize);
  let dx = sx - u.centreX;
  let dy = sy - u.centreY;
  let ix = u.centreX + dx * u.cosT - dy * u.sinT;
  let iy = u.centreY + dx * u.sinT + dy * u.cosT;
  let ex = ix + f32(u.margin);
  let ey = iy + f32(u.margin);

  var value = 0.0;
  if (channel < u.mddfChannels) {
    let srcRe = table[channel * 4u];
    let srcIm = table[channel * 4u + 1u];
    let mode = table[channel * 4u + 2u];
    let isIm = table[channel * 4u + 3u];
    if (mode == 0u) {
      value = sampleField(ey, ex, srcRe);
    } else {
      let re = sampleField(ey, ex, srcRe);
      let im = sampleField(ey, ex, srcIm);
      let angle = f32(mode) * u.theta;
      let c = cos(angle); let s = sin(angle);
      if (isIm == 1u) { value = re * s + im * c; }
      else { value = re * c - im * s; }
    }
    value = (value - norm[channel * 2u]) / norm[channel * 2u + 1u];
  } else if (i32(channel) == u.solidChannel) {
    let ry = clamp(i32(round(ey)), 0, i32(u.gy) - 1);
    let rx = clamp(i32(round(ex)), 0, i32(u.gx) - 1);
    let roofLayer = round(height[u32(ry) * u.gx + u32(rx)] / ${wgslFloat(VERTICAL_SPACING_M)});
    value = select(0.0, 1.0, roofLayer > u.queryLayer);
  } else {
    let which = i32(channel) - u.coordChannel;
    if (which == 0) { value = (sy - u.centreY) / u.coordYScale; }
    else if (which == 1) { value = u.coordZ; }
    else { value = (sx - u.coordXCentre) / u.coordXScale; }
  }
  out[g] = ${store === "f16" ? "f16(value)" : "value"};
}`;

export class MddfField {
  /**
   * @param {import("./context.js").Context} context
   * @param {Float32Array} layerMap roof layer count per city cell, [ny, nx]
   * @param {number} margin expansion per side, in cells
   */
  constructor(context, layerMap, ny, nx, margin, store = "f32") {
    this.context = context;
    this.store = store;
    this.ny = ny;
    this.nx = nx;
    this.margin = margin;
    this.gy = ny + 2 * margin;
    this.gx = nx + 2 * margin;
    this.chunk = Math.min(16384, this.gy * this.gx);

    // Expanded roof heights in metres; outside the city is open ground.
    const heights = new Float32Array(this.gy * this.gx);
    for (let y = 0; y < ny; y++) {
      const src = y * nx;
      const dst = (y + margin) * this.gx + margin;
      for (let x = 0; x < nx; x++) {
        heights[dst + x] = layerMap[src + x] * VERTICAL_SPACING_M;
      }
    }
    this.heightBuffer = context.upload(heights, "mddf.height");

    const cosT = new Float32Array(ANGLE_COUNT);
    const sinT = new Float32Array(ANGLE_COUNT);
    for (let a = 0; a < ANGLE_COUNT; a++) {
      const theta = 2 * Math.PI * a / ANGLE_COUNT;
      cosT[a] = Math.cos(theta);
      sinT[a] = Math.sin(theta);
    }
    this.cosBuffer = context.upload(cosT, "mddf.cos");
    this.sinBuffer = context.upload(sinT, "mddf.sin");
    this.rawBuffer = context.storage(
      SIGNAL_COUNT * this.chunk * ANGLE_COUNT * 4, "mddf.raw");
    this.fieldBuffer = context.storage(
      this.gy * this.gx * FIELD_CHANNELS * 4, "mddf.field");

    this.tracePipeline = context.pipeline(TRACE);
    this.fourierPipeline = context.pipeline(FOURIER);
    this.assemblePipeline = context.pipeline(assembleShader(store));
    this.currentLayer = -1;
  }

  /** Trace and reduce the whole expanded grid for one query height. */
  async compute(layerIndex) {
    const { context } = this;
    const queryHeight = (layerIndex + 0.5) * VERTICAL_SPACING_M;
    const pixels = this.gy * this.gx;
    for (let offset = 0; offset < pixels; offset += this.chunk) {
      const count = Math.min(this.chunk, pixels - offset);
      const uniform = context.uniform([
        ["u", this.gy], ["u", this.gx], ["u", this.chunk], ["u", offset],
        ["u", count], ["u", 0], ["u", 0], ["u", 0],
        ["f", queryHeight], ["f", 0], ["f", 0], ["f", 0],
      ]);
      const encoder = context.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      context.dispatch(pass, this.tracePipeline, [
        this.heightBuffer, this.cosBuffer, this.sinBuffer,
        this.rawBuffer, uniform,
      ], count * ANGLE_COUNT);
      context.dispatch(pass, this.fourierPipeline, [
        this.rawBuffer, this.fieldBuffer, uniform,
      ], count * 4, 64);
      pass.end();
      context.device.queue.submit([encoder.finish()]);
    }
    await context.device.queue.onSubmittedWorkDone();
    context.releaseScratch();
    this.currentLayer = layerIndex;
  }

  /**
   * Build the model input tensor for a batch of obliquely placed patches.
   * @param {Float32Array} origins wind-frame [sx0, sy0] pairs
   * @param {object} plan channel table, normalization and coord constants
   */
  assemble(encoder, origins, plan, theta, layerIndex, patch, outputBuffer) {
    const { context } = this;
    const batch = origins.length / 2;
    const originBuffer = context.upload(origins, "assemble.origins");
    const uniform = context.uniform([
      ["u", batch], ["u", plan.channels], ["u", plan.mddfChannels], ["u", patch],
      ["u", this.gy], ["u", this.gx], ["u", this.margin], ["u", 0],
      ["i", plan.solidChannel], ["i", plan.coordChannel], ["i", 0], ["i", 0],
      ["f", Math.cos(theta)], ["f", Math.sin(theta)], ["f", theta],
      ["f", layerIndex],
      ["f", (this.ny - 1) / 2], ["f", (this.nx - 1) / 2],
      ["f", plan.coordYScale],
      ["f", (layerIndex - plan.coordZCentre) / plan.coordZScale],
      ["f", plan.coordXCentre], ["f", plan.coordXScale], ["f", 0], ["f", 0],
    ]);
    const pass = encoder.beginComputePass();
    context.dispatch(pass, this.assemblePipeline, [
      this.fieldBuffer, this.heightBuffer, originBuffer,
      plan.tableBuffer, plan.normBuffer, outputBuffer, uniform,
    ], batch * plan.channels * patch * patch);
    pass.end();
    return originBuffer;
  }

  destroy() {
    for (const buffer of [this.heightBuffer, this.cosBuffer, this.sinBuffer,
                          this.rawBuffer, this.fieldBuffer]) {
      this.context.free(buffer);
    }
  }
}
