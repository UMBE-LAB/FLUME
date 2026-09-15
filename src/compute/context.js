// WebGPU device, buffer accounting and pipeline cache.
// Ported from the validated feasibility prototype (feasibility_test/gpu.js).

export class Context {
  constructor(adapter, device, useF16) {
    this.adapter = adapter;
    this.device = device;
    this.useF16 = useF16;
    this.liveBytes = 0;
    this.peakBytes = 0;
    this.pipelines = new Map();
    this.scratch = [];
  }

  static async create() {
    if (!navigator.gpu) throw new Error(
      "WebGPU unavailable. Use desktop Chrome or Edge over https:// or " +
      "http://localhost (a secure context is required).");
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("No WebGPU adapter available.");
    const f16 = adapter.features.has("shader-f16");
    const device = await adapter.requestDevice({
      requiredFeatures: f16 ? ["shader-f16"] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize, 1 << 30),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
      },
    });
    return new Context(adapter, device, f16);
  }

  async describe() {
    let info = this.adapter.info;
    if (!info && this.adapter.requestAdapterInfo) {
      info = await this.adapter.requestAdapterInfo();
    }
    const parts = info
      ? [info.vendor, info.architecture, info.device, info.description]
      : [];
    const name = parts.filter(Boolean).join(" ") || "unidentified GPU";
    return `${name}${this.useF16 ? " · f16" : " · f32"}`;
  }

  buffer(bytes, usage, label) {
    const size = Math.max(4, Math.ceil(bytes / 4) * 4);
    const buffer = this.device.createBuffer({ size, usage, label });
    buffer._tracked = size;
    this.liveBytes += size;
    this.peakBytes = Math.max(this.peakBytes, this.liveBytes);
    return buffer;
  }

  storage(bytes, label, extra = 0) {
    return this.buffer(
      bytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC | extra,
      label);
  }

  free(buffer) {
    if (buffer && buffer._tracked) {
      this.liveBytes -= buffer._tracked;
      buffer._tracked = 0;
      buffer.destroy();
    }
  }

  upload(array, label) {
    const buffer = this.storage(array.byteLength, label);
    this.write(buffer, array);
    return buffer;
  }

  write(buffer, array) {
    // writeBuffer requires a multiple of four bytes; odd-length f16 or
    // 16-bit index arrays need padding up to the allocated size.
    if (array.byteLength % 4) {
      const padded = new Uint8Array(Math.ceil(array.byteLength / 4) * 4);
      padded.set(new Uint8Array(
        array.buffer, array.byteOffset, array.byteLength));
      this.device.queue.writeBuffer(buffer, 0, padded);
    } else {
      this.device.queue.writeBuffer(buffer, 0, array);
    }
  }

  // Small per-dispatch uniform block, recycled after each submission.
  // `fields` is a list of [kind, value] with kind in {"u","i","f"}; the type
  // must be explicit because a float that happens to be integral (cos 0 = 1.0)
  // would otherwise be written with integer bits.
  uniform(fields) {
    const size = Math.ceil(fields.length / 4) * 16;
    const data = new ArrayBuffer(size);
    const asU32 = new Uint32Array(data);
    const asI32 = new Int32Array(data);
    const asF32 = new Float32Array(data);
    fields.forEach(([kind, value], index) => {
      if (kind === "u") asU32[index] = value;
      else if (kind === "i") asI32[index] = value;
      else asF32[index] = value;
    });
    const buffer = this.device.createBuffer({
      size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, data);
    this.scratch.push(buffer);
    return buffer;
  }

  releaseScratch() {
    for (const buffer of this.scratch) buffer.destroy();
    this.scratch = [];
  }

  // Every uniform declared in a shader must actually be read: an unread
  // binding is stripped from an "auto" layout and the bind group then fails.
  pipeline(code) {
    if (!this.pipelines.has(code)) {
      const module = this.device.createShaderModule({ code });
      this.pipelines.set(code, this.device.createComputePipeline({
        layout: "auto", compute: { module, entryPoint: "main" },
      }));
    }
    return this.pipelines.get(code);
  }

  dispatch(pass, pipeline, buffers, threads, workgroup = 256) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => (
        { binding, resource: { buffer } })),
    }));
    // A single dispatch dimension is capped (65535 on common hardware), so
    // spill the excess into the second dimension; the shaders index linearly
    // through num_workgroups.
    const groups = Math.max(1, Math.ceil(threads / workgroup));
    const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
    pass.dispatchWorkgroups(
      Math.min(groups, limit), Math.ceil(groups / limit));
  }

  async readFloat32(buffer, count) {
    const bytes = count * 4;
    const staging = this.buffer(
      bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, "staging");
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, staging.size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(
      staging.getMappedRange().slice(0, bytes));
    staging.unmap();
    this.free(staging);
    return result;
  }
}

export function toHalf(values) {
  const out = new Uint16Array(values.length);
  const scratch = new Float32Array(1);
  const bits = new Uint32Array(scratch.buffer);
  for (let i = 0; i < values.length; i++) {
    scratch[0] = values[i];
    const x = bits[0];
    const sign = (x >>> 16) & 0x8000;
    const exponent = (x >>> 23) & 0xff;
    let mantissa = x & 0x7fffff;
    if (exponent === 0xff) { out[i] = sign | 0x7c00 | (mantissa ? 1 : 0); continue; }
    const e = exponent - 127 + 15;
    if (e >= 31) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }
      mantissa |= 0x800000;
      const shift = 14 - e;
      let half = mantissa >>> shift;
      if ((mantissa >>> (shift - 1)) & 1) half += 1;
      out[i] = sign | half;
      continue;
    }
    let half = (e << 10) | (mantissa >>> 13);
    if (mantissa & 0x1000) half += 1;
    out[i] = sign | half;
  }
  return out;
}

export function fromHalf(half) {
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) {
    const h = half[i];
    const sign = (h & 0x8000) ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;
    if (exponent === 0) out[i] = sign * mantissa * 2 ** -24;
    else if (exponent === 31) out[i] = mantissa ? NaN : sign * Infinity;
    else out[i] = sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return out;
}
