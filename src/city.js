// City height maps: built-in assets and user uploads share one decoder.
//
// Upload contract (see WEB_DEMO_HANDOFF.md section 3): PNG-8 grayscale, any
// size, one pixel = one 4 m cell, pixel value = number of 1.5 m building
// layers.  Row = Y (cross-wind), column = X (streamwise, 0 degrees along +X).

export const VERTICAL_SPACING_M = 1.5;
export const HORIZONTAL_SPACING_M = 4.0;
export const MAXIMUM_LAYERS = 160;

async function decode(blob, label) {
  const bitmap = await createImageBitmap(blob);
  // Read the dimensions before closing: an ImageBitmap reports 0 x 0 once
  // closed, which silently produces an empty city.
  const nx = bitmap.width;
  const ny = bitmap.height;
  const canvas = new OffscreenCanvas(nx, ny);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, nx, ny);
  bitmap.close();

  const cells = nx * ny;
  const layers = new Float32Array(cells);
  let maximum = 0;
  let colored = false;
  for (let i = 0; i < cells; i++) {
    const r = data[i * 4];
    if (r !== data[i * 4 + 1] || r !== data[i * 4 + 2]) colored = true;
    layers[i] = r;
    if (r > maximum) maximum = r;
  }
  const notes = [];
  if (colored) {
    notes.push("image is not grayscale; the red channel was used");
  }
  if (maximum > MAXIMUM_LAYERS) {
    throw new Error(
      `${label}: pixel value ${maximum} exceeds ${MAXIMUM_LAYERS} layers. ` +
      "Values must be building layer counts, not metres or centimetres.");
  }
  if (!cells) throw new Error(`${label}: decoded to an empty image`);
  return {
    label, layers, ny, nx, notes,
    maximumLayers: maximum,
    maximumHeight: maximum * VERTICAL_SPACING_M,
    extent: [ny * HORIZONTAL_SPACING_M, nx * HORIZONTAL_SPACING_M],
  };
}

export async function loadManifest(base = "assets/cities") {
  const response = await fetch(`${base}/manifest.json`);
  if (!response.ok) throw new Error("city manifest is unavailable");
  const manifest = await response.json();
  manifest.base = base;
  return manifest;
}

export async function loadBuiltIn(manifest, id, signal) {
  const entry = manifest.cities.find((city) => city.id === id);
  if (!entry) throw new Error(`unknown city ${id}`);
  const response = await fetch(`${manifest.base}/${entry.file}`, { signal });
  if (!response.ok) throw new Error(`cannot fetch ${entry.file}`);
  return decode(await response.blob(), entry.label);
}

export async function loadUpload(file) {
  if (!/\.png$/i.test(file.name)) {
    throw new Error("upload a PNG height map (see the format note)");
  }
  return decode(file, file.name);
}

/** Bilinear sample of the layer map, for building outlines and probes. */
export function layerAt(city, iy, ix) {
  const y = Math.min(city.ny - 1, Math.max(0, Math.round(iy)));
  const x = Math.min(city.nx - 1, Math.max(0, Math.round(ix)));
  return city.layers[y * city.nx + x];
}

/**
 * A small square tile of a height map, for the city picker.
 *
 * The tile is always square and the map is drawn to fit inside it, centred.
 * Today every built-in city is square so nothing is letterboxed, but maps of
 * other proportions are coming, and fitting rather than stretching is what
 * keeps the picker correct when they arrive: a wide city will read as wide.
 *
 * Size is deliberately NOT encoded in the drawing.  Drawing every city at one
 * metres-per-pixel would put the 3.0 km city at full width and the 1.2 km ones
 * at 40 % of it -- 16 % of the area -- and at this tile size that destroys the
 * street pattern, which is the only thing that tells the cities apart.  The
 * extent is reported in words instead, and felt through the scale bar.
 */
export async function thumbnailFrom(blob, size) {
  const probe = await createImageBitmap(blob);
  const fit = Math.min(size / probe.width, size / probe.height);
  const width = Math.max(1, Math.round(probe.width * fit));
  const height = Math.max(1, Math.round(probe.height * fit));
  probe.close();
  const small = await createImageBitmap(blob, {
    resizeWidth: width, resizeHeight: height, resizeQuality: "high",
  });
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const left = Math.round((size - width) / 2);
  const top = Math.round((size - height) / 2);
  context.drawImage(small, left, top);
  small.close();

  // Recolour: the raw values are layer counts, and a typical city tops out
  // around 40 of the 160 possible, so drawn as grey levels the whole tile is
  // nearly black.  Normalising against the tile's own maximum is what makes
  // the street pattern visible.
  const image = context.getImageData(left, top, width, height);
  const data = image.data;
  let peak = 0;
  for (let i = 0; i < data.length; i += 4) {
    if (data[i] > peak) peak = data[i];
  }
  const ground = [0x23, 0x26, 0x2d];      // the same grey as the 3-D ground
  const low = [0x4a, 0x51, 0x5c];
  const high = [0xe2, 0xe8, 0xef];
  for (let i = 0; i < data.length; i += 4) {
    const layers = data[i];
    if (layers < 1 || peak < 1) {
      data[i] = ground[0]; data[i + 1] = ground[1]; data[i + 2] = ground[2];
    } else {
      const t = Math.min(1, layers / peak);
      for (let c = 0; c < 3; c++) data[i + c] = low[c] + (high[c] - low[c]) * t;
    }
    data[i + 3] = 255;
  }
  context.putImageData(image, left, top);
  return canvas;
}

/** The same tile for a built-in city, fetched by id. */
export async function thumbnailOf(manifest, id, size) {
  const entry = manifest.cities.find((city) => city.id === id);
  if (!entry) throw new Error(`unknown city ${id}`);
  const response = await fetch(`${manifest.base}/${entry.file}`);
  if (!response.ok) throw new Error(`cannot fetch ${entry.file}`);
  return thumbnailFrom(await response.blob(), size);
}
