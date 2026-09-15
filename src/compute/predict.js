// Oblique patch tiling, batched inference and overlap-add reconstruction.
//
// The city never rotates.  For wind direction theta the patches are cut with
// their axes along the wind, in cross-wind columns marching downwind; columns
// do not have to line up with each other, because the Hann blend normalizes by
// its own weight sum.  The prediction therefore lives on an oblique pixel grid,
// which is what gets displayed -- no second resampling back to city axes.
//
// Columns march downwind rather than strips running alongside the wind so that
// submission order is streamwise order: the progressive fill then sweeps the
// way the wind blows, which is also why patch index order must stay the order
// the patches are computed in.

import { MddfField, VERTICAL_SPACING_M } from "./mddf.js";
import { Fno } from "./fno.js";

// Gather form of overlap-add: every output cell sums the patches covering it.
// A scatter would need floating-point atomics, which WebGPU does not provide.
const wgslFloat = (value) => value.toFixed(6);

const blendShader = (store) => `${store === "f16" ? "enable f16;\n" : ""}
struct P {
  gridH: u32, gridW: u32, patchSize: u32, channels: u32,
  columnCount: u32, patchCount: u32, readyCount: u32, pad0: u32,
  rectY0: u32, rectX0: u32, rectH: u32, rectW: u32,
  columnStride: f32, minimumWeight: f32, pad1: f32, pad2: f32,
};
@group(0) @binding(0) var<storage, read> patches: array<${store}>;
// Cross-wind origin of each patch, in blend coordinates.
@group(0) @binding(1) var<storage, read> origins: array<i32>;
// Per column: first patch index, patch count, streamwise origin.
@group(0) @binding(2) var<storage, read> columns: array<i32>;
@group(0) @binding(3) var<storage, read_write> field: array<f32>;
@group(0) @binding(4) var<storage, read_write> weight: array<f32>;
@group(0) @binding(5) var<uniform> u: P;

fn hann(index: u32, size: u32) -> f32 {
  let t = 6.283185307179586 * f32(index) / f32(size - 1u);
  return u.minimumWeight + (1.0 - u.minimumWeight) * (0.5 - 0.5 * cos(t));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  // The dispatch covers a rectangle of the wind-frame grid, not the whole
  // grid: a progressive fill re-blends only the cells the newest patches
  // touch, so the total work stays proportional to the patch area no matter
  // how often the display is refreshed.
  if (g >= u.rectH * u.rectW) { return; }
  let sy = i32(u.rectY0 + g / u.rectW);
  let sx = i32(u.rectX0 + g % u.rectW);
  let cell = u32(sy) * u.gridW + u32(sx);
  let patchSize = i32(u.patchSize);
  let area = u.patchSize * u.patchSize;

  var sums = array<f32, 8>();
  var total = 0.0;
  // Columns are regularly spaced, so only a couple can cover this cell.
  let firstColumn = max(0,
    i32(floor((f32(sx) - f32(patchSize) + 1.0) / max(u.columnStride, 1.0e-6))));
  let lastColumn = min(i32(u.columnCount) - 1,
    i32(floor(f32(sx) / max(u.columnStride, 1.0e-6))));
  for (var k = firstColumn; k <= lastColumn; k++) {
    let columnBase = u32(k) * 3u;
    let first = columns[columnBase];
    let count = columns[columnBase + 1u];
    let px = sx - columns[columnBase + 2u];
    if (px < 0 || px >= patchSize) { continue; }
    for (var j = 0; j < count; j++) {
      let index = first + j;
      // Patches are submitted in index order, so everything from readyCount
      // onwards has not been computed yet and must not enter the sum.
      if (u32(index) >= u.readyCount) { break; }
      let py = sy - origins[index];
      if (py < 0 || py >= patchSize) { continue; }
      let w = hann(u32(py), u.patchSize) * hann(u32(px), u.patchSize);
      let base = u32(index) * u.channels * area + u32(py) * u.patchSize + u32(px);
      for (var c = 0u; c < u.channels; c++) {
        sums[c] += w * f32(patches[base + c * area]);
      }
      total += w;
    }
  }
  for (var c = 0u; c < u.channels; c++) {
    field[cell * u.channels + c] = sums[c];
  }
  weight[cell] = total;
}`;

// Divide by the accumulated weight, convert to physical units, mark solids
// and flag pixels that fall outside the city domain.
// `pack` also writes the rectangle out in compact rect-local order as
// [c0..c4, solid, coverage] per cell, so a progressive redraw reads back only
// the cells it is about to paint instead of the whole field.  Coverage is the
// accumulated blend weight over the weight the cell will end up with, which is
// exactly "how finished is this cell" and is used as the display alpha.
const FINALIZE = `
struct P {
  gridH: u32, gridW: u32, channels: u32, pack: u32,
  rectY0: u32, rectX0: u32, rectH: u32, rectW: u32,
  syStart: i32, sxStart: i32, pad0: i32, pad1: i32,
  cosT: f32, sinT: f32, centreY: f32, centreX: f32,
  gy: f32, gx: f32, margin: f32, queryLayer: f32,
  cityY: f32, cityX: f32, pad2: f32, pad3: f32,
};
@group(0) @binding(0) var<storage, read_write> field: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> centre: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read> height: array<f32>;
@group(0) @binding(5) var<storage, read_write> solid: array<f32>;
@group(0) @binding(6) var<storage, read> finalWeight: array<f32>;
@group(0) @binding(7) var<storage, read_write> packed: array<f32>;
@group(0) @binding(8) var<uniform> u: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * 256u;
  if (g >= u.rectH * u.rectW) { return; }
  let ly = g / u.rectW;
  let lx = g % u.rectW;
  let cell = (u.rectY0 + ly) * u.gridW + (u.rectX0 + lx);
  let w = weight[cell];
  let inverse = select(0.0, 1.0 / w, w > 1.0e-6);
  for (var c = 0u; c < u.channels; c++) {
    field[cell * u.channels + c] =
      field[cell * u.channels + c] * inverse * scale[c] + centre[c];
  }
  let sx = f32(i32(u.rectX0 + lx) + u.sxStart);
  let sy = f32(i32(u.rectY0 + ly) + u.syStart);
  let dx = sx - u.centreX;
  let dy = sy - u.centreY;
  let ix = u.centreX + dx * u.cosT - dy * u.sinT;
  let iy = u.centreY + dx * u.sinT + dy * u.cosT;
  var flag = 0.0;
  if (ix < -0.5 || iy < -0.5 || ix > u.cityX - 0.5 || iy > u.cityY - 0.5
      || w <= 1.0e-6) {
    flag = 2.0;
  } else {
    let ry = clamp(i32(round(iy + u.margin)), 0, i32(u.gy) - 1);
    let rx = clamp(i32(round(ix + u.margin)), 0, i32(u.gx) - 1);
    let layer = round(height[u32(ry) * u32(u.gx) + u32(rx)]
                      / ${wgslFloat(VERTICAL_SPACING_M)});
    if (layer > u.queryLayer) { flag = 1.0; }
  }
  solid[cell] = flag;
  if (u.pack == 1u) {
    let fw = finalWeight[cell];
    let coverage = select(0.0, clamp(w / fw, 0.0, 1.0), fw > 1.0e-6);
    let out = g * (u.channels + 2u);
    for (var c = 0u; c < u.channels; c++) {
      packed[out + c] = field[cell * u.channels + c];
    }
    packed[out + u.channels] = flag;
    packed[out + u.channels + 1u] = coverage;
  }
}`;

/** Smallest rectangle containing both, or the other one if either is null. */
function unionRect(a, b) {
  if (!a) return b;
  if (!b) return a;
  const y0 = Math.min(a.y0, b.y0);
  const x0 = Math.min(a.x0, b.x0);
  return {
    y0, x0,
    h: Math.max(a.y0 + a.h, b.y0 + b.h) - y0,
    w: Math.max(a.x0 + a.w, b.x0 + b.w) - x0,
  };
}

/** Convex clip of the rotated city rectangle by a horizontal wind-frame band. */
function bandExtent(corners, syLow, syHigh) {
  let low = Infinity;
  let high = -Infinity;
  const consider = (x) => { low = Math.min(low, x); high = Math.max(high, x); };
  for (let i = 0; i < 4; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % 4];
    if (a[1] >= syLow && a[1] <= syHigh) consider(a[0]);
    for (const edge of [syLow, syHigh]) {
      if ((a[1] - edge) * (b[1] - edge) < 0) {
        const t = (edge - a[1]) / (b[1] - a[1]);
        consider(a[0] + t * (b[0] - a[0]));
      }
    }
  }
  return low <= high ? [low, high] : null;
}

export class Predictor {
  constructor(context, metadata, blob, options = {}) {
    this.context = context;
    this.metadata = metadata;
    this.patch = metadata.inference.patch[0];
    this.stride = Math.max(
      1, Math.round(this.patch * (1 - metadata.inference.overlap)));
    this.batchSize = options.batchSize || 16;
    this.fno = new Fno(context, metadata, blob, this.batchSize);
    this.blendPipeline = context.pipeline(blendShader(this.fno.store));
    this.finalizePipeline = context.pipeline(FINALIZE);
    this.plan = this.buildPlan();
    this.city = null;
  }

  /** Channel table and normalization for the fused assemble pass. */
  buildPlan() {
    const { context, metadata } = this;
    const features = metadata.features;
    const perSignal = metadata.mddf.channels_per_signal;
    const modeCount = metadata.mddf.modes_per_signal;
    const indices = features.mddf_source_indices;
    const table = new Uint32Array(indices.length * 4);
    const norm = new Float32Array(indices.length * 2);
    indices.forEach((source, channel) => {
      const signal = Math.floor(source / perSignal);
      const within = source % perSignal;
      const isIm = within > modeCount ? 1 : 0;
      const mode = within === 0 ? 0 : (isIm ? within - modeCount : within);
      table.set([
        signal * perSignal + mode,
        signal * perSignal + (mode === 0 ? 0 : modeCount + mode),
        mode, isIm,
      ], channel * 4);
      norm[channel * 2] = features.mddf_center[channel];
      norm[channel * 2 + 1] = features.mddf_scale[channel];
    });
    const coords = metadata.coordinates;
    return {
      channels: metadata.model.input_channels,
      mddfChannels: indices.length,
      solidChannel: features.use_building_input ? indices.length : -1,
      coordChannel: indices.length + (features.use_building_input ? 1 : 0),
      coordYScale: coords.y_scale,
      coordZCentre: coords.z_center, coordZScale: coords.z_scale,
      coordXCentre: coords.x_center, coordXScale: coords.x_scale,
      tableBuffer: context.upload(table, "plan.table"),
      normBuffer: context.upload(norm, "plan.norm"),
    };
  }

  /** Load a city height map (roof layer counts, row-major [ny, nx]). */
  setCity(layerMap, ny, nx) {
    if (this.city) this.city.field.destroy();
    // A patch that touches the domain can reach one patch diagonal beyond it.
    const margin = Math.ceil(this.patch * Math.SQRT2);
    // A 750-cell city needs a 93 MiB feature buffer, close to the 128 MiB
    // default binding limit; fail with a readable message rather than a raw
    // validation error.
    const needed = (ny + 2 * margin) * (nx + 2 * margin) * 28 * 4;
    const limit = this.context.device.limits.maxStorageBufferBindingSize;
    if (needed > limit) {
      throw new Error(
        `city ${nx} x ${ny} needs a ${(needed / 2 ** 20).toFixed(0)} MiB ` +
        `feature buffer, but this GPU allows ` +
        `${(limit / 2 ** 20).toFixed(0)} MiB`);
    }
    const field = new MddfField(
      this.context, layerMap, ny, nx, margin, this.fno.store);
    this.city = { field, ny, nx, layerMap, margin, layerIndex: -1 };
    return { margin, gy: field.gy, gx: field.gx };
  }

  /** Trace MDDF for one height; the result serves every wind direction. */
  async setHeight(layerIndex) {
    if (!this.city) throw new Error("no city loaded");
    if (this.city.layerIndex === layerIndex) return false;
    await this.city.field.compute(layerIndex);
    this.city.layerIndex = layerIndex;
    return true;
  }

  /** Wind-frame tiling: rows at a fixed stride, each covering its own span. */
  tiling(theta) {
    const { ny, nx } = this.city;
    const centreY = (ny - 1) / 2;
    const centreX = (nx - 1) / 2;
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    const corners = [[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]]
      .map(([x, y]) => {
        const dx = x - centreX;
        const dy = y - centreY;
        return [centreX + dx * cos + dy * sin, centreY - dx * sin + dy * cos];
      });
    const syMin = Math.floor(Math.min(...corners.map((c) => c[1])));
    const syMax = Math.ceil(Math.max(...corners.map((c) => c[1])));
    const sxMin = Math.floor(Math.min(...corners.map((c) => c[0])));
    const sxMax = Math.ceil(Math.max(...corners.map((c) => c[0])));

    const spanX = sxMax - sxMin + 1;
    const columnCount = Math.max(
      1, Math.ceil((spanX - this.patch) / this.stride) + 1);
    const columnStride =
      columnCount > 1 ? (spanX - this.patch) / (columnCount - 1) : 0;

    // bandExtent clips by its second coordinate, so feeding it corners with x
    // and y exchanged clips the rotated city by a vertical band and returns the
    // cross-wind range that band covers.
    const flipped = corners.map(([x, y]) => [y, x]);
    const origins = [];
    const columns = [];
    for (let k = 0; k < columnCount; k++) {
      const sx0 = Math.round(sxMin + k * columnStride);
      const extent = bandExtent(flipped, sx0, sx0 + this.patch - 1)
                  || [syMin, syMin + this.patch - 1];
      const low = Math.floor(extent[0]);
      const spanY = Math.max(
        this.patch, Math.ceil(extent[1]) - low + 1);
      const count = Math.max(
        1, Math.ceil((spanY - this.patch) / this.stride) + 1);
      const step = count > 1 ? (spanY - this.patch) / (count - 1) : 0;
      columns.push([origins.length / 2, count, sx0 - sxMin]);
      for (let j = 0; j < count; j++) {
        origins.push(sx0, Math.round(low + j * step));
      }
    }
    return {
      origins: new Float32Array(origins), columns, columnCount, columnStride,
      syMin, sxMin, gridH: syMax - syMin + 1, gridW: sxMax - sxMin + 1,
      patchCount: origins.length / 2,
    };
  }

  /**
   * Bounding box, in wind-frame blend coordinates, of the cells that patches
   * [from, to) can touch.  This is the region a progressive redraw has to
   * recompute after that group of patches lands.
   */
  patchRect(layout, from, to) {
    let y0 = Infinity;
    let x0 = Infinity;
    let y1 = -Infinity;
    let x1 = -Infinity;
    for (const [first, count, columnX] of layout.columns) {
      for (let j = 0; j < count; j++) {
        const index = first + j;
        if (index < from || index >= to) continue;
        const py = layout.origins[index * 2 + 1] - layout.syMin;
        y0 = Math.min(y0, py);
        x0 = Math.min(x0, columnX);
        y1 = Math.max(y1, py + this.patch);
        x1 = Math.max(x1, columnX + this.patch);
      }
    }
    if (!Number.isFinite(y0)) return null;
    y0 = Math.max(0, y0);
    x0 = Math.max(0, x0);
    y1 = Math.min(layout.gridH, y1);
    x1 = Math.min(layout.gridW, x1);
    return { y0, x0, h: Math.max(0, y1 - y0), w: Math.max(0, x1 - x0) };
  }

  /**
   * Predict one slice for the current city and height at wind direction theta.
   * Values are physical five-channel data on the oblique wind-frame grid.
   */
  /**
   * Predict the slice at wind direction theta.
   *
   * With `options.shouldDraw` / `options.onDraw` the display is filled in as
   * the patches land.  Each redraw re-blends only the rectangle the newest
   * patches touch, so the extra work is proportional to the patch area and
   * does not grow with the redraw rate: a cell is recomputed once per group of
   * patches covering it, whether the display refreshes three times or three
   * hundred.  The closing full-grid pass is the same one a non-progressive run
   * does, so the returned slice is identical either way.
   */
  async predict(theta, options = {}) {
    const { context, city } = this;
    const layout = this.tiling(theta);
    const channels = this.metadata.model.output_channels;
    const cells = layout.gridH * layout.gridW;
    const area = this.patch * this.patch;
    const started = performance.now();
    const streaming = Boolean(options.shouldDraw && options.onDraw);
    const packStride = channels + 2;

    const patchBuffer = context.storage(
      layout.patchCount * channels * area * this.fno.elementBytes,
      "predict.patches");

    // Origins in blend space are relative to the wind-frame grid origin.  The
    // blend indexes patches by column, so what it needs per patch is the
    // cross-wind origin; the streamwise one is a property of the column.
    const originY = new Int32Array(layout.patchCount);
    for (let i = 0; i < layout.patchCount; i++) {
      originY[i] = layout.origins[i * 2 + 1] - layout.syMin;
    }
    const originBuffer = context.upload(originY, "blend.origins");
    const columnBuffer = context.upload(
      new Int32Array(layout.columns.flat()), "blend.columns");
    const fieldBuffer = context.storage(cells * channels * 4, "blend.field");
    const weightBuffer = context.storage(cells * 4, "blend.weight");
    const solidBuffer = context.storage(cells * 4, "blend.solid");
    // Only the progressive path needs the coverage denominator and the compact
    // read-back staging area.  A plain run still has to bind something in those
    // slots, but it must be a distinct buffer: binding one buffer twice in a
    // group where either binding is writable is not allowed.  The shader guards
    // every access to them behind `pack`, so a placeholder is never read.
    const finalWeight = context.storage(
      streaming ? cells * 4 : 4, "blend.finalWeight");
    const packedBuffer = context.storage(
      streaming ? cells * packStride * 4 : 4, "blend.packed");
    const target = this.metadata.target;
    const centreBuffer = context.upload(
      new Float32Array(target.center), "final.centre");
    const scaleBuffer = context.upload(
      new Float32Array(target.scale), "final.scale");

    const whole = { y0: 0, x0: 0, h: layout.gridH, w: layout.gridW };

    const blend = (encoder, rect, readyCount, weightTarget) => {
      const pass = encoder.beginComputePass();
      context.dispatch(pass, this.blendPipeline, [
        patchBuffer, originBuffer, columnBuffer, fieldBuffer, weightTarget,
        context.uniform([
          ["u", layout.gridH], ["u", layout.gridW], ["u", this.patch],
          ["u", channels], ["u", layout.columnCount], ["u", layout.patchCount],
          ["u", readyCount], ["u", 0],
          ["u", rect.y0], ["u", rect.x0], ["u", rect.h], ["u", rect.w],
          ["f", layout.columnStride],
          ["f", this.metadata.inference.blend_window_minimum],
          ["f", 0], ["f", 0],
        ]),
      ], rect.h * rect.w);
      pass.end();
    };

    const finalize = (encoder, rect, pack) => {
      const pass = encoder.beginComputePass();
      context.dispatch(pass, this.finalizePipeline, [
        fieldBuffer, weightBuffer, centreBuffer, scaleBuffer,
        city.field.heightBuffer, solidBuffer, finalWeight, packedBuffer,
        context.uniform([
          ["u", layout.gridH], ["u", layout.gridW], ["u", channels],
          ["u", pack ? 1 : 0],
          ["u", rect.y0], ["u", rect.x0], ["u", rect.h], ["u", rect.w],
          ["i", layout.syMin], ["i", layout.sxMin], ["i", 0], ["i", 0],
          ["f", Math.cos(theta)], ["f", Math.sin(theta)],
          ["f", (city.ny - 1) / 2], ["f", (city.nx - 1) / 2],
          ["f", city.field.gy], ["f", city.field.gx],
          ["f", city.margin], ["f", city.layerIndex],
          ["f", city.ny], ["f", city.nx], ["f", 0], ["f", 0],
        ]),
      ], rect.h * rect.w);
      pass.end();
    };

    if (streaming) {
      // Coverage denominator: run the blend with every patch marked ready
      // while the patch buffer is still zero-filled, and keep only the weight
      // it accumulates.  WebGPU zero-initializes buffers, so the field sums
      // this writes are zeros and the next blend overwrites them anyway.
      const encoder = context.device.createCommandEncoder();
      blend(encoder, whole, layout.patchCount, finalWeight);
      context.device.queue.submit([encoder.finish()]);
      options.onStart?.({
        gridH: layout.gridH, gridW: layout.gridW,
        originY: layout.syMin, originX: layout.sxMin,
        theta, patchCount: layout.patchCount,
      });
    }

    // Encoding a batch costs the CPU microseconds, so without a brake the loop
    // queues every batch inside one frame and the wall clock says nothing about
    // how far the GPU has got: the first read-back then waits for everything at
    // once and the whole field appears in one step.  Letting the CPU lead by a
    // couple of batches keeps the GPU fed while making elapsed time mean GPU
    // progress, which is what the refresh gate is timing.
    const RUN_AHEAD = 2;
    const inFlight = [];

    let dirty = null;
    for (let offset = 0; offset < layout.patchCount; offset += this.batchSize) {
      const count = Math.min(this.batchSize, layout.patchCount - offset);
      const encoder = context.device.createCommandEncoder();
      const scratch = city.field.assemble(
        encoder, layout.origins.subarray(offset * 2, (offset + count) * 2),
        this.plan, theta, city.layerIndex, this.patch, this.fno.inputBuffer);
      const output = this.fno.encode(encoder, count);
      encoder.copyBufferToBuffer(
        output, 0, patchBuffer,
        offset * channels * area * this.fno.elementBytes,
        count * channels * area * this.fno.elementBytes);

      // The redraw rides in the same encoder as the batch that feeds it:
      // queue order guarantees the patches are written before the blend reads
      // them, and it saves a submission round trip per refresh.
      let paint = null;
      if (streaming) {
        dirty = unionRect(dirty, this.patchRect(layout, offset, offset + count));
        // The last batch always refreshes.  Otherwise whatever it covers stays
        // half-faded until the closing full render pops it in, which is the one
        // discontinuity a viewer would actually notice.
        const last = offset + count >= layout.patchCount;
        if (dirty && dirty.h > 0 && dirty.w > 0
            && (last || options.shouldDraw())) {
          paint = dirty;
          dirty = null;
          blend(encoder, paint, offset + count, weightBuffer);
          finalize(encoder, paint, true);
        }
      }
      context.device.queue.submit([encoder.finish()]);
      context.free(scratch);
      if (streaming) {
        inFlight.push(context.device.queue.onSubmittedWorkDone());
        if (inFlight.length > RUN_AHEAD) await inFlight.shift();
      }

      if (paint) {
        const packed = await context.readFloat32(
          packedBuffer, paint.h * paint.w * packStride);
        options.onDraw({
          x0: paint.x0, y0: paint.y0, width: paint.w, height: paint.h,
          stride: packStride, channels, theta, values: packed,
          readyCount: offset + count, patchCount: layout.patchCount,
        });
      }
    }

    // Closing full-grid pass.  The progressive rectangles left those cells in
    // physical units; this rewrites the raw sums for every cell and converts
    // once, so the outcome does not depend on which rectangles were drawn.
    const encoder = context.device.createCommandEncoder();
    blend(encoder, whole, layout.patchCount, weightBuffer);
    finalize(encoder, whole, false);
    context.device.queue.submit([encoder.finish()]);

    const values = await context.readFloat32(fieldBuffer, cells * channels);
    const solid = await context.readFloat32(solidBuffer, cells);
    const weights = await context.readFloat32(weightBuffer, cells);
    const elapsed = performance.now() - started;

    // Separate the two ways a cell can end up unusable: never covered by a
    // patch, or covered but carrying a non-finite value from the operator.
    let uncovered = 0;
    let nonFinite = 0;
    for (let cell = 0; cell < cells; cell++) {
      if (solid[cell] === 2) continue;
      if (weights[cell] <= 1.0e-6) { uncovered += 1; continue; }
      for (let c = 0; c < channels; c++) {
        if (!Number.isFinite(values[cell * channels + c])) { nonFinite += 1; break; }
      }
    }

    for (const buffer of [patchBuffer, fieldBuffer, weightBuffer, solidBuffer,
                          originBuffer, columnBuffer, centreBuffer, scaleBuffer,
                          finalWeight, packedBuffer]) {
      context.free(buffer);
    }
    context.releaseScratch();

    return {
      values, solid, theta,
      height: layout.gridH, width: layout.gridW,
      originY: layout.syMin, originX: layout.sxMin,
      patchCount: layout.patchCount,
      channels: target.channel_names,
      milliseconds: elapsed,
      uncovered, nonFinite, insideCells: cells,
    };
  }

  destroy() {
    if (this.city) this.city.field.destroy();
    this.fno.destroy();
    this.context.free(this.plan.tableBuffer);
    this.context.free(this.plan.normBuffer);
  }
}
