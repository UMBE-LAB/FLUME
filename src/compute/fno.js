// 2-D FLUME operator on WebGPU, driven entirely by the exported metadata.
//
// The spectral convolution keeps only a few low modes, so the transform is
// evaluated as small dense DFT products rather than an FFT: exact, expressible
// as plain matrix work, and free of any browser FFT dependency.  The staged
// formulation was checked against PyTorch (max output error 2.6e-6).
//
// Spectral weights arrive as a Tucker core plus factor matrices and are
// contracted into the dense stack once, on the GPU, at load time.

import { Context, toHalf, fromHalf } from "./context.js";

const COMMON = `
fn erf_approx(x: f32) -> f32 {
  let s = sign(x);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return s * y;
}
fn gelu(x: f32) -> f32 {
  return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476));
}
`;

// Contract mode `mode` of a five-dimensional complex tensor with a factor.
const CONTRACT = `
struct P { d0: u32, d1: u32, d2: u32, d3: u32,
           d4: u32, mode: u32, outer: u32, rank: u32,
           strideMode: u32, blockAbove: u32, total: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> facRe: array<f32>;
@group(0) @binding(3) var<storage, read> facIm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(5) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(6) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.total) { return; }
  // Split the flat output index around the contracted mode.
  let inner = g % u.strideMode;
  let a = (g / u.strideMode) % u.outer;
  let above = g / (u.strideMode * u.outer);
  var accRe = 0.0;
  var accIm = 0.0;
  for (var r = 0u; r < u.rank; r++) {
    let source = (above * u.rank + r) * u.strideMode + inner;
    let xr = inRe[source];
    let xi = inIm[source];
    let fr = facRe[a * u.rank + r];
    let fi = facIm[a * u.rank + r];
    accRe += xr * fr - xi * fi;
    accIm += xr * fi + xi * fr;
  }
  outRe[g] = accRe;
  outIm[g] = accIm;
}`;

// InstanceNorm, one workgroup per (sample, channel).  It takes no parameters,
// so the dispatch size is the only input and there is no uniform binding.
//
// Two passes on purpose: the single-pass identity E[x^2] - mean^2 loses all
// significance when a channel has a large mean and a small spread, and the
// cancellation error can push the variance below zero, at which point
// inverseSqrt yields NaN for the whole channel.  Summing squared deviations
// cannot go negative.
//
// The epsilon is a parameter because the spectral branch stores its result
// pre-scaled: InstanceNorm is invariant to a constant factor k on its input
// provided epsilon is replaced by epsilon / k^2, which makes the scaling exact
// rather than an approximation.
function instanceNormShader(inStore, outStore, area, epsilon = 1.0e-5) {
  const enable = (inStore === "f16" || outStore === "f16") ? "enable f16;\n" : "";
  const read = (expr) => inStore === "f16" ? `f32(${expr})` : expr;
  const write = (expr) => outStore === "f16" ? `f16(${expr})` : expr;
  return `${enable}
@group(0) @binding(0) var<storage, read> X: array<${inStore}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${outStore}>;
var<workgroup> partial: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${area}u;
  var s = 0.0;
  for (var i = lid.x; i < ${area}u; i += 256u) { s += ${read("X[base + i]")}; }
  partial[lid.x] = s;
  workgroupBarrier();
  var stride = 128u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] += partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let mean = partial[0] / ${area}.0;
  workgroupBarrier();

  var q = 0.0;
  for (var i = lid.x; i < ${area}u; i += 256u) {
    let d = ${read("X[base + i]")} - mean;
    q += d * d;
  }
  partial[lid.x] = q;
  workgroupBarrier();
  stride = 128u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] += partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let inverse = inverseSqrt(max(partial[0] / ${area}.0, 0.0) + ${epsilon.toExponential(12)});
  for (var i = lid.x; i < ${area}u; i += 256u) {
    Y[base + i] = ${write("(" + read("X[base + i]") + " - mean) * inverse")};
  }
}`;
}

// The spectral result peaks near 8.2e4 over the tested cities and heights,
// beyond the 65504 limit of f16.  Storing it pre-divided by this power of two
// keeps 25x of headroom while staying far above the subnormal range; the
// following InstanceNorm undoes it exactly through its epsilon.
export const SPECTRAL_SCALE = 32;

function shaders(spec) {
  const { store, patch, modes1, modes2 } = spec;
  const area = patch * patch;
  const rows = 2 * modes1;
  const enable = store === "f16" ? "enable f16;\n" : "";
  const load = (expr) => store === "f16" ? `f32(${expr})` : expr;
  const save = (expr) => store === "f16" ? `f16(${expr})` : expr;

  return {
    conv1x1: `${enable}${COMMON}
struct P { batch: u32, cin: u32, cout: u32, act: u32 };
@group(0) @binding(0) var<storage, read> X: array<${store}>;
@group(0) @binding(1) var<storage, read> W: array<${store}>;
@group(0) @binding(2) var<storage, read> B: array<${store}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${store}>;
@group(0) @binding(4) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.batch * u.cout * ${area}u) { return; }
  let p = g % ${area}u;
  let o = (g / ${area}u) % u.cout;
  let b = g / (${area}u * u.cout);
  var acc = ${load("B[o]")};
  let xBase = b * u.cin * ${area}u + p;
  let wBase = o * u.cin;
  for (var i = 0u; i < u.cin; i++) {
    acc += ${load("W[wBase + i]")} * ${load("X[xBase + i * " + area + "u]")};
  }
  if (u.act == 1u) { acc = gelu(acc); }
  Y[g] = ${save("acc")};
}`,

    gate: `${enable}${COMMON}
struct P { batch: u32, mid: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> G: array<${store}>;
@group(0) @binding(1) var<storage, read_write> H: array<${store}>;
@group(0) @binding(2) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.batch * u.mid * ${area}u) { return; }
  let p = g % ${area}u;
  let m = (g / ${area}u) % u.mid;
  let b = g / (${area}u * u.mid);
  let value = ${load("G[(b * 2u * u.mid + m) * " + area + "u + p]")};
  let gateValue = ${load("G[(b * 2u * u.mid + u.mid + m) * " + area + "u + p]")};
  H[g] = ${save("gelu(value * (1.0 / (1.0 + exp(-gateValue))))")};
}`,

    instanceNorm: instanceNormShader(store, store, area),
    instanceNormScaled: instanceNormShader(
      store, store, area, 1.0e-5 / (SPECTRAL_SCALE * SPECTRAL_SCALE)),

    // Forward DFT along x, truncated to modes2 columns.
    forwardColumns: `${enable}
struct P { batch: u32, channels: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> X: array<${store}>;
@group(0) @binding(1) var<storage, read> tableCos: array<f32>;
@group(0) @binding(2) var<storage, read> tableSin: array<f32>;
@group(0) @binding(3) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(4) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(5) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.batch * u.channels * ${patch * modes2}u) { return; }
  let k = g % ${modes2}u;
  let y = (g / ${modes2}u) % ${patch}u;
  let bc = g / ${patch * modes2}u;
  var accRe = 0.0; var accIm = 0.0;
  let base = bc * ${area}u + y * ${patch}u;
  for (var x = 0u; x < ${patch}u; x++) {
    let v = ${load("X[base + x]")};
    accRe += v * tableCos[x * ${modes2}u + k];
    accIm += v * tableSin[x * ${modes2}u + k];
  }
  outRe[g] = accRe; outIm[g] = accIm;
}`,

    // Complex matrix product along y; used for both the forward reduction
    // (patch -> 2*modes1 rows) and the inverse expansion.
    rowMix: `
struct P { batch: u32, channels: u32, rows: u32, inner: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> tableRe: array<f32>;
@group(0) @binding(3) var<storage, read> tableIm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(5) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(6) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.batch * u.channels * u.rows * ${modes2}u) { return; }
  let k = g % ${modes2}u;
  let r = (g / ${modes2}u) % u.rows;
  let bc = g / (${modes2}u * u.rows);
  var accRe = 0.0; var accIm = 0.0;
  let inBase = bc * u.inner * ${modes2}u + k;
  let tableBase = r * u.inner;
  for (var j = 0u; j < u.inner; j++) {
    let ar = inRe[inBase + j * ${modes2}u];
    let ai = inIm[inBase + j * ${modes2}u];
    let cr = tableRe[tableBase + j];
    let ci = tableIm[tableBase + j];
    accRe += ar * cr - ai * ci;
    accIm += ar * ci + ai * cr;
  }
  outRe[g] = accRe; outIm[g] = accIm;
}`,

    // Channel mixing by the layer's spectral weights, per retained mode.
    spectralMix: `
struct P { batch: u32, channels: u32, stackBase: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> wRe: array<f32>;
@group(0) @binding(3) var<storage, read> wIm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(5) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(6) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  let block = ${rows * modes2}u;
  if (g >= u.batch * u.channels * block) { return; }
  let k = g % ${modes2}u;
  let s = (g / ${modes2}u) % ${rows}u;
  let o = (g / block) % u.channels;
  let b = g / (block * u.channels);
  let half = select(0u, 1u, s >= ${modes1}u);
  let sp = s - half * ${modes1}u;
  var accRe = 0.0; var accIm = 0.0;
  let modeArea = ${modes1 * modes2}u;
  let wBase = ((u.stackBase + half) * u.channels * u.channels + o) * modeArea
            + sp * ${modes2}u + k;
  let xBase = (b * u.channels * ${rows}u + s) * ${modes2}u + k;
  for (var i = 0u; i < u.channels; i++) {
    let xr = inRe[xBase + i * block];
    let xi = inIm[xBase + i * block];
    let wr = wRe[wBase + i * u.channels * modeArea];
    let wi = wIm[wBase + i * u.channels * modeArea];
    accRe += xr * wr - xi * wi;
    accIm += xr * wi + xi * wr;
  }
  outRe[g] = accRe; outIm[g] = accIm;
}`,

    // Inverse DFT along x back to the patch grid (real output).
    inverseColumns: `${enable}
struct P { batch: u32, channels: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> tableCos: array<f32>;
@group(0) @binding(3) var<storage, read> tableSin: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<${store}>;
@group(0) @binding(5) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.batch * u.channels * ${area}u) { return; }
  let p = g % ${area}u;
  let x = p % ${patch}u;
  let y = p / ${patch}u;
  let bc = g / ${area}u;
  var acc = 0.0;
  let base = (bc * ${patch}u + y) * ${modes2}u;
  for (var k = 0u; k < ${modes2}u; k++) {
    acc += inRe[base + k] * tableCos[k * ${patch}u + x]
         + inIm[base + k] * tableSin[k * ${patch}u + x];
  }
  Y[g] = ${save("acc * " + (1 / SPECTRAL_SCALE))};
}`,

    residual: `${enable}${COMMON}
struct P { total: u32, act: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> A: array<${store}>;
@group(0) @binding(1) var<storage, read> B: array<${store}>;
@group(0) @binding(2) var<storage, read_write> Y: array<${store}>;
@group(0) @binding(3) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.total) { return; }
  var acc = ${load("A[g]")} + ${load("B[g]")} + ${load("Y[g]")};
  if (u.act == 1u) { acc = gelu(acc); }
  Y[g] = ${save("acc")};
}`,
  };
}

function dftTables(patch, modes1, modes2) {
  const rows = 2 * modes1;
  const ky = [];
  for (let s = 0; s < modes1; s++) ky.push(s);
  for (let s = patch - modes1; s < patch; s++) ky.push(s);

  const forwardCos = new Float32Array(patch * modes2);
  const forwardSin = new Float32Array(patch * modes2);
  for (let x = 0; x < patch; x++) {
    for (let k = 0; k < modes2; k++) {
      const t = 2 * Math.PI * x * k / patch;
      forwardCos[x * modes2 + k] = Math.cos(t);
      forwardSin[x * modes2 + k] = -Math.sin(t);
    }
  }
  const rowRe = new Float32Array(rows * patch);
  const rowIm = new Float32Array(rows * patch);
  for (let s = 0; s < rows; s++) {
    for (let y = 0; y < patch; y++) {
      const t = 2 * Math.PI * ky[s] * y / patch;
      rowRe[s * patch + y] = Math.cos(t);
      rowIm[s * patch + y] = -Math.sin(t);
    }
  }
  const inverseRowRe = new Float32Array(patch * rows);
  const inverseRowIm = new Float32Array(patch * rows);
  for (let y = 0; y < patch; y++) {
    for (let s = 0; s < rows; s++) {
      const t = 2 * Math.PI * ky[s] * y / patch;
      inverseRowRe[y * rows + s] = Math.cos(t) / patch;
      inverseRowIm[y * rows + s] = Math.sin(t) / patch;
    }
  }
  // Real reconstruction from the retained half-spectrum; the Nyquist bin is
  // never inside the retained range, so every non-zero mode doubles.
  const inverseColCos = new Float32Array(modes2 * patch);
  const inverseColSin = new Float32Array(modes2 * patch);
  for (let k = 0; k < modes2; k++) {
    const amplitude = (k === 0 ? 1 : 2) / patch;
    for (let x = 0; x < patch; x++) {
      const t = 2 * Math.PI * k * x / patch;
      inverseColCos[k * patch + x] = amplitude * Math.cos(t);
      inverseColSin[k * patch + x] = -amplitude * Math.sin(t);
    }
  }
  return { forwardCos, forwardSin, rowRe, rowIm, inverseRowRe, inverseRowIm,
           inverseColCos, inverseColSin };
}

export class Fno {
  /**
   * @param {Context} context
   * @param {object} metadata exported model metadata
   * @param {ArrayBuffer} blob  weights.bin
   * @param {number} maxBatch   patches per submission
   */
  constructor(context, metadata, blob, maxBatch) {
    const model = metadata.model;
    this.context = context;
    this.metadata = metadata;
    this.width = model.width;
    this.layers = model.layers;
    this.inputChannels = model.input_channels;
    this.outputChannels = model.output_channels;
    this.patch = metadata.inference.patch[0];
    this.modes1 = model.modes[0];
    this.modes2 = model.modes[1];
    this.maxBatch = maxBatch;
    this.store = context.useF16 ? "f16" : "f32";
    this.elementBytes = this.store === "f16" ? 2 : 4;
    if (metadata.inference.patch[0] !== metadata.inference.patch[1]) {
      throw new Error("only square patches are supported");
    }

    const spec = { store: this.store, patch: this.patch,
                   modes1: this.modes1, modes2: this.modes2 };
    this.pipelines = {};
    for (const [name, code] of Object.entries(shaders(spec))) {
      this.pipelines[name] = context.pipeline(code);
    }
    this.contractPipeline = context.pipeline(CONTRACT);

    const read = (name) => {
      const entry = metadata.tensors[name];
      if (!entry) throw new Error(`missing tensor ${name}`);
      return fromHalf(new Uint16Array(blob, entry.offset, entry.count));
    };
    this.read = read;

    this.tables = {};
    for (const [name, data] of
         Object.entries(dftTables(this.patch, this.modes1, this.modes2))) {
      this.tables[name] = context.upload(data, `fno.table.${name}`);
    }

    this.weights = {};
    const upload = (key, name) => {
      const values = read(name);
      this.weights[key] = this.store === "f16"
        ? context.upload(toHalf(values), `fno.${key}`)
        : context.upload(values, `fno.${key}`);
    };
    upload("p1w", "p.mlp1.weight"); upload("p1b", "p.mlp1.bias");
    upload("p2w", "p.mlp2.weight"); upload("p2b", "p.mlp2.bias");
    upload("q1w", "q.mlp1.weight"); upload("q1b", "q.mlp1.bias");
    upload("q2w", "q.mlp2.weight"); upload("q2b", "q.mlp2.bias");
    for (let l = 0; l < this.layers; l++) {
      upload(`m1w${l}`, `mlp_layers.${l}.mlp1.weight`);
      upload(`m1b${l}`, `mlp_layers.${l}.mlp1.bias`);
      upload(`m2w${l}`, `mlp_layers.${l}.mlp2.weight`);
      upload(`m2b${l}`, `mlp_layers.${l}.mlp2.bias`);
      upload(`lw${l}`, `w_layers.${l}.weight`);
      upload(`lb${l}`, `w_layers.${l}.bias`);
    }

    this.buildSpectralWeights();
    this.allocate(maxBatch);
  }

  /** Contract the Tucker core with its factors into the dense weight stack. */
  buildSpectralWeights() {
    const { context, metadata } = this;
    const tucker = metadata.tucker;
    let dims = tucker.core_shape.slice();
    let re = context.upload(this.read(`${tucker.core}.re`), "tucker.core.re");
    let im = context.upload(this.read(`${tucker.core}.im`), "tucker.core.im");

    for (const factor of tucker.factors) {
      const [outer, rank] = factor.shape;
      if (dims[factor.mode] !== rank) {
        throw new Error(`factor ${factor.name} does not match the core`);
      }
      const next = dims.slice();
      next[factor.mode] = outer;
      const total = next.reduce((a, b) => a * b, 1);
      const strideMode = next.slice(factor.mode + 1)
        .reduce((a, b) => a * b, 1);
      const outRe = context.storage(total * 4, "tucker.re");
      const outIm = context.storage(total * 4, "tucker.im");
      const facRe = context.upload(this.read(`${factor.name}.re`), "tucker.f");
      const facIm = context.upload(this.read(`${factor.name}.im`), "tucker.f");
      const uniform = context.uniform([
        ["u", next[0]], ["u", next[1]], ["u", next[2]], ["u", next[3]],
        ["u", next[4]], ["u", factor.mode], ["u", outer], ["u", rank],
        ["u", strideMode], ["u", 0], ["u", total], ["u", 0],
      ]);
      const encoder = context.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      context.dispatch(pass, this.contractPipeline,
        [re, im, facRe, facIm, outRe, outIm, uniform], total);
      pass.end();
      context.device.queue.submit([encoder.finish()]);
      context.free(re); context.free(im);
      context.free(facRe); context.free(facIm);
      re = outRe; im = outIm; dims = next;
    }
    context.releaseScratch();
    const expected = [2 * this.layers, this.width, this.width,
                      this.modes1, this.modes2];
    if (dims.join() !== expected.join()) {
      throw new Error(`spectral stack is ${dims} but ${expected} was expected`);
    }
    this.spectralRe = re;
    this.spectralIm = im;
  }

  allocate(batch) {
    const { context } = this;
    const area = this.patch * this.patch;
    const C = this.width;
    const rows = 2 * this.modes1;
    const activation = (channels, label) => context.storage(
      batch * channels * area * this.elementBytes, label);
    const complex = (count, label) =>
      context.storage(batch * C * count * 4, label);
    this.buffers = {
      input: activation(this.inputChannels, "fno.input"),
      x: activation(C, "fno.x"),
      t1: activation(C, "fno.t1"),
      t2: activation(C, "fno.t2"),
      gates: activation(4 * C, "fno.gates"),
      hidden: activation(2 * C, "fno.hidden"),
      spectral: activation(C, "fno.spectral"),
      local: activation(C, "fno.local"),
      wide: activation(2 * C, "fno.wide"),
      output: activation(this.outputChannels, "fno.output"),
      xRe: complex(this.patch * this.modes2, "fno.xRe"),
      xIm: complex(this.patch * this.modes2, "fno.xIm"),
      yRe: complex(rows * this.modes2, "fno.yRe"),
      yIm: complex(rows * this.modes2, "fno.yIm"),
      zRe: complex(rows * this.modes2, "fno.zRe"),
      zIm: complex(rows * this.modes2, "fno.zIm"),
      wRe: complex(this.patch * this.modes2, "fno.wRe"),
      wIm: complex(this.patch * this.modes2, "fno.wIm"),
    };
  }

  get inputBuffer() { return this.buffers.input; }

  /** Bytes the activation buffers need for a given batch, for budgeting. */
  allocationBytes(batch) {
    const area = this.patch * this.patch;
    const C = this.width;
    const rows = 2 * this.modes1;
    const channels = this.inputChannels + 5 * C + 4 * C + 2 * C + 2 * C
                   + this.outputChannels;
    const complex = C * (2 * this.patch * this.modes2 * 2
                         + 4 * rows * this.modes2) * 4;
    return batch * (channels * area * this.elementBytes + complex);
  }

  /** Encode a forward pass over `batch` patches already present in `input`. */
  encode(encoder, batch) {
    const { context, buffers: b, pipelines: p, tables: t } = this;
    const C = this.width;
    const area = this.patch * this.patch;
    const rows = 2 * this.modes1;
    const pass = encoder.beginComputePass();
    const run = (pipeline, resources, values, threads, group = 256) => {
      context.dispatch(pass, pipeline,
        [...resources, context.uniform(values)], threads, group);
    };
    const norm = (source, target, pipeline = p.instanceNorm) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, context.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: source } },
                  { binding: 1, resource: { buffer: target } }],
      }));
      pass.dispatchWorkgroups(batch * C);
    };

    run(p.conv1x1, [b.input, this.weights.p1w, this.weights.p1b, b.wide],
        [["u", batch], ["u", this.inputChannels], ["u", 2 * C], ["u", 1]],
        batch * 2 * C * area);
    run(p.conv1x1, [b.wide, this.weights.p2w, this.weights.p2b, b.x],
        [["u", batch], ["u", 2 * C], ["u", C], ["u", 0]], batch * C * area);

    for (let l = 0; l < this.layers; l++) {
      norm(b.x, b.t1);
      run(p.forwardColumns, [b.t1, t.forwardCos, t.forwardSin, b.xRe, b.xIm],
          [["u", batch], ["u", C], ["u", 0], ["u", 0]],
          batch * C * this.patch * this.modes2);
      run(p.rowMix, [b.xRe, b.xIm, t.rowRe, t.rowIm, b.yRe, b.yIm],
          [["u", batch], ["u", C], ["u", rows], ["u", this.patch]],
          batch * C * rows * this.modes2);
      run(p.spectralMix,
          [b.yRe, b.yIm, this.spectralRe, this.spectralIm, b.zRe, b.zIm],
          [["u", batch], ["u", C], ["u", 2 * l], ["u", 0]],
          batch * C * rows * this.modes2);
      run(p.rowMix,
          [b.zRe, b.zIm, t.inverseRowRe, t.inverseRowIm, b.wRe, b.wIm],
          [["u", batch], ["u", C], ["u", this.patch], ["u", rows]],
          batch * C * this.patch * this.modes2);
      run(p.inverseColumns,
          [b.wRe, b.wIm, t.inverseColCos, t.inverseColSin, b.t2],
          [["u", batch], ["u", C], ["u", 0], ["u", 0]], batch * C * area);
      norm(b.t2, b.t1, p.instanceNormScaled);
      run(p.conv1x1,
          [b.t1, this.weights[`m1w${l}`], this.weights[`m1b${l}`], b.gates],
          [["u", batch], ["u", C], ["u", 4 * C], ["u", 0]],
          batch * 4 * C * area);
      run(p.gate, [b.gates, b.hidden],
          [["u", batch], ["u", 2 * C], ["u", 0], ["u", 0]],
          batch * 2 * C * area);
      run(p.conv1x1,
          [b.hidden, this.weights[`m2w${l}`], this.weights[`m2b${l}`],
           b.spectral],
          [["u", batch], ["u", 2 * C], ["u", C], ["u", 0]], batch * C * area);
      run(p.conv1x1,
          [b.x, this.weights[`lw${l}`], this.weights[`lb${l}`], b.local],
          [["u", batch], ["u", C], ["u", C], ["u", 0]], batch * C * area);
      run(p.residual, [b.spectral, b.local, b.x],
          [["u", batch * C * area], ["u", l < this.layers - 1 ? 1 : 0],
           ["u", 0], ["u", 0]], batch * C * area);
    }

    run(p.conv1x1, [b.x, this.weights.q1w, this.weights.q1b, b.wide],
        [["u", batch], ["u", C], ["u", 2 * C], ["u", 1]], batch * 2 * C * area);
    run(p.conv1x1, [b.wide, this.weights.q2w, this.weights.q2b, b.output],
        [["u", batch], ["u", 2 * C], ["u", this.outputChannels], ["u", 0]],
        batch * this.outputChannels * area);
    pass.end();
    return b.output;
  }

  destroy() {
    const { context } = this;
    for (const buffer of Object.values(this.buffers)) context.free(buffer);
    for (const buffer of Object.values(this.weights)) context.free(buffer);
    for (const buffer of Object.values(this.tables)) context.free(buffer);
    context.free(this.spectralRe);
    context.free(this.spectralIm);
  }
}
