// Particle streamlines: a screen-space trail buffer, composited through the
// slice plane so it is still occluded by the buildings.
//
// The trail is an accumulation buffer at canvas resolution, so it is sharp at
// any zoom and has no resolution parameter.  Each frame draws only the one
// segment a particle just travelled; the streak is what the buffer has not yet
// faded away.  Because the buffer is screen-space it is stale the moment the
// camera moves, so it is cleared then and the streaks grow back.
//
// The buffer is not blitted over the scene -- it is sampled by the slice plane's
// own overlay mesh at gl_FragCoord, which means depth testing and the city
// clipping planes apply to it exactly as they do to the field underneath.
//
// Particles are advected on the CPU in wind-frame grid cells, which is where
// the model's u and v already live, so nothing is rotated.  A couple of
// thousand particles cost nothing there, and it keeps every decision -- respawn,
// bounds, density -- in plain code instead of packed into a texture.

import * as THREE from "../vendor/three.module.js";

// Particle lifetime as a multiple of the trail length; see respawn().
const LIFE_PER_TRAIL = [1.5, 3.0];

// Buffer capacity, not a setting: the vertex arrays are allocated once at this
// size and `maxParticles` slides underneath it.
export const PARTICLE_CAPACITY = 8000;

// Every number that shapes the look, held per mode.  The white streaks sit on
// top of a coloured field and the coloured ones replace it, so they are being
// judged against completely different backgrounds and there is no reason for
// them to want the same width, density or trail length.
export const DEFAULTS = {
  white: {
    pixelsPerParticle: 680,   // screen pixels per particle; lower = denser
    maxParticles: 1200,
    minParticles: 150,
    lineWidth: 2.5,           // pixels
    trailSeconds: 3.5,        // how long a streak lasts, end to end
    solidFraction: 0.45,      // the last of its life is where it fades
    trailCutoff: 0.03,        // below this the texel is wiped outright
    timeScale: 30,            // flow seconds per real second
    strength: 1.0,            // overall opacity of the composited trail
    lineOpacity: 1.0,         // opacity a fresh segment is drawn with
  },
  tinted: {
    pixelsPerParticle: 340,
    maxParticles: 2400,
    minParticles: 300,
    lineWidth: 2.5,
    trailSeconds: 3.5,
    solidFraction: 0.45,
    trailCutoff: 0.03,
    timeScale: 30,
    strength: 1.0,
    lineOpacity: 1.0,
  },
};

/** What the tuning panel builds its sliders from. */
export const PARAMS = [
  { key: "pixelsPerParticle", label: "density (px per particle, lower = denser)",
    min: 40, max: 1600, step: 10, primary: true },
  { key: "maxParticles", label: "max particles", min: 50, max: PARTICLE_CAPACITY, step: 50 },
  { key: "minParticles", label: "min particles", min: 0, max: 3000, step: 25 },
  { key: "lineWidth", label: "line width (px)", min: 0.5, max: 24, step: 0.25, primary: true },
  { key: "trailSeconds", label: "trail length (s)", min: 0.2, max: 15, step: 0.1, primary: true },
  { key: "solidFraction", label: "solid fraction (fade starts after)", min: 0.02, max: 1, step: 0.01 },
  { key: "trailCutoff", label: "trail cutoff", min: 0, max: 0.3, step: 0.005 },
  { key: "timeScale", label: "flow speed (flow s per real s)", min: 1, max: 150, step: 1, primary: true },
  { key: "strength", label: "trail opacity", min: 0, max: 1, step: 0.01 },
  { key: "lineOpacity", label: "segment opacity", min: 0, max: 1, step: 0.01 },
];

const QUAD_VERTEX = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

// Linear decay, by time rather than by frame so the streak keeps its length
// when the frame rate drops.  Not exponential: exp() sheds most of its opacity
// in the first fraction of the life, which leaves nearly the whole streak
// translucent, and a translucent bright colour over a dark scene reads as a
// glow rather than as a stroke of paint.  Straight down to zero, in a fixed
// time, with a cutoff so nothing is left stuck at one 8-bit level for ever.
const FADE_FRAGMENT = `
precision mediump float;
uniform sampler2D previous;
uniform float drop;
uniform float cutoff;
varying vec2 vUv;
void main() {
  vec4 before = texture2D(previous, vUv);
  float alpha = before.a - drop;
  if (alpha < cutoff || before.a == 0.0) {
    gl_FragColor = vec4(0.0);
  } else {
    // The buffer holds premultiplied colour (see LINE_FRAGMENT), so colour
    // and opacity fall together.  Lowering only the opacity would leave the
    // colour too bright for it, and un-premultiplying later would overshoot.
    gl_FragColor = before * (alpha / before.a);
  }
}`;

// Colour rides on the vertices.  In the white mode every segment is handed the
// same colour; in the coloured mode it is the field's own ramp sampled where
// the particle stood, so an old streak keeps the colours it passed through --
// the accumulation buffer stores them, nothing extra is needed to remember.
const LINE_VERTEX = `
attribute vec2 segment;
attribute vec3 tint;
attribute vec4 ends;          // the segment's two endpoints, in buffer pixels
varying vec3 vTint;
varying vec4 vEnds;
void main() {
  vTint = tint;
  vEnds = ends;
  gl_Position = vec4(segment, 0.0, 1.0);
}`;

// The quad is only a bounding box; the shape inside it is a capsule -- the set
// of points within half a line width of the segment.  That is what gives the
// head a round cap instead of the flat end of a rectangle, rounds the joint
// between one frame's segment and the next, and antialiases the edge, all from
// one distance test.
const LINE_FRAGMENT = `
precision mediump float;
uniform float opacity;
uniform float halfWidth;
varying vec3 vTint;
varying vec4 vEnds;
void main() {
  vec2 point = gl_FragCoord.xy;
  vec2 along = vEnds.zw - vEnds.xy;
  float t = clamp(dot(point - vEnds.xy, along) / max(dot(along, along), 1e-6),
                  0.0, 1.0);
  float distance = length(point - (vEnds.xy + along * t));
  float edge = 1.0 - smoothstep(halfWidth - 1.0, halfWidth, distance);
  if (edge <= 0.0) discard;
  // The output is straight colour, but what lands in the buffer is not: the
  // SRC_ALPHA blend multiplies it by alpha on the way in, so the buffer holds
  // premultiplied colour.  That is exactly premultiplied "over" onto what is
  // already there, so this pass needs nothing else -- but the passes reading
  // the buffer must treat it as premultiplied.  Treating it as plain colour
  // is what painted a dark grey ring around every line: an edge pixel half
  // covered by white was stored as 0.5 grey and drawn back as 0.5 grey.
  gl_FragColor = vec4(vTint, opacity * edge);
}`;

// Sampling the trail at gl_FragCoord is what keeps it screen-space -- and
// therefore pixel-sharp -- while still being drawn as part of the plane, so the
// depth buffer and the clipping planes do their usual job.
const OVERLAY_VERTEX = `
#include <clipping_planes_pars_vertex>
void main() {
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;

const OVERLAY_FRAGMENT = `
#include <clipping_planes_pars_fragment>
precision mediump float;
uniform sampler2D trail;
uniform vec2 resolution;
uniform float strength;
uniform float solid;
void main() {
  #include <clipping_planes_fragment>
  vec4 texel = texture2D(trail, gl_FragCoord.xy / resolution);
  if (texel.a <= 0.004) discard;
  // Opaque for most of a streak's life, fading only over the last stretch of
  // it.  Without this the stored alpha is the whole profile and the streak is
  // translucent everywhere except at its very head.
  float alpha = min(1.0, texel.a / solid) * strength;
  // No colour-space encode here.  The tint was looked up in the same sRGB
  // ramp the legend is drawn from and carried through unchanged, so it is
  // already a display colour; encoding it again lifts every value and drains
  // the saturation -- turbo's darkest step leaves as 48,18,59 and arrives as
  // 119,76,130.  The slice can afford the encode because its palette texture
  // is tagged sRGB and the GPU decodes it on the way in.
  // The buffer is premultiplied; divide once to get the line's own colour
  // back.  texel.a is above 0.004 here, so the division is safe.
  gl_FragColor = vec4(texel.rgb / texel.a, alpha);
}`;

export class ParticleFlow {
  constructor(renderer) {
    this.renderer = renderer;
    this.enabled = false;
    this.field = null;
    this.particles = [];
    this.viewKey = "";
    this.width = 1;
    this.height = 1;
    this.stats = { target: 0, drawn: 0, placed: 0, bounds: null };
    // One settings block per mode; `mode` selects which one every frame reads.
    this.settings = {
      white: { ...DEFAULTS.white }, tinted: { ...DEFAULTS.tinted },
    };
    this.mode = "white";
    this.palette = null;      // set by setTinting when colouring by the field
    this.tintRange = null;
    this.lastValue = 0;
    // How much of the prediction has arrived, and how far downwind it reaches.
    // Both are needed while a field is still filling in: the particle count is
    // per unit of flowable area, not per unit of grid, and spawning into cells
    // that have not arrived yet only wastes attempts.
    this.ready = 1;
    this.readyX = Infinity;
    this.extent = { x0: 0, y0: 0, x1: 0, y1: 0 };
    this.degenerate = false;

    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
    this.quad.frustumCulled = false;
    this.quadScene = new THREE.Scene().add(this.quad);
    this.targets = null;

    this.fade = new THREE.ShaderMaterial({
      vertexShader: QUAD_VERTEX, fragmentShader: FADE_FRAGMENT,
      uniforms: { previous: { value: null }, drop: { value: 0.02 },
                  cutoff: { value: DEFAULTS.white.trailCutoff } },
      depthTest: false, depthWrite: false, side: THREE.DoubleSide,
    });

    this.vertices = new Float32Array(PARTICLE_CAPACITY * 12);
    this.tints = new Float32Array(PARTICLE_CAPACITY * 18);
    this.ends = new Float32Array(PARTICLE_CAPACITY * 24);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("segment",
      new THREE.BufferAttribute(this.vertices, 2).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("tint",
      new THREE.BufferAttribute(this.tints, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("ends",
      new THREE.BufferAttribute(this.ends, 4).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("position",
      new THREE.BufferAttribute(new Float32Array(PARTICLE_CAPACITY * 18), 3));
    this.lineGeometry = geometry;
    this.lines = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
      vertexShader: LINE_VERTEX, fragmentShader: LINE_FRAGMENT,
      uniforms: { opacity: { value: DEFAULTS.white.lineOpacity },
                  halfWidth: { value: DEFAULTS.white.lineWidth * 0.5 } },
      depthTest: false, depthWrite: false, transparent: true,
      // Not decoration: any side other than DoubleSide makes three enable
      // back-face culling, and a segment quad's winding follows the direction
      // the particle happens to be moving.  With the flow mostly one way, that
      // culls nearly every segment and the trail buffer stays empty.
      side: THREE.DoubleSide,
    }));
    this.lines.frustumCulled = false;
    this.lineScene = new THREE.Scene().add(this.lines);

    this.material = new THREE.ShaderMaterial({
      vertexShader: OVERLAY_VERTEX, fragmentShader: OVERLAY_FRAGMENT,
      uniforms: { trail: { value: null },
                  resolution: { value: new THREE.Vector2(1, 1) },
                  strength: { value: DEFAULTS.white.strength },
                  solid: { value: DEFAULTS.white.solidFraction } },
      transparent: true, depthWrite: false, side: THREE.DoubleSide,
      clipping: true,
    });
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), this.material);
    this.mesh.visible = false;

    this.planeSize = new THREE.Vector2(1, 1);
    this.plane = new THREE.Plane();
    this.ray = new THREE.Ray();
    this.scratch = new THREE.Vector3();
    this.matrix = new THREE.Matrix4();
  }

  /**
   * Sized in drawing-buffer pixels, not CSS pixels: the overlay samples this
   * texture at gl_FragCoord, which counts device pixels, so on any display
   * with a pixel ratio above one a CSS-sized buffer is sampled off its edge.
   */
  setResolution(width, height) {
    if (width === this.width && height === this.height) return;
    this.width = Math.max(1, Math.round(width));
    this.height = Math.max(1, Math.round(height));
    for (const target of this.targets || []) target.dispose();
    this.targets = [0, 1].map(() =>
      new THREE.WebGLRenderTarget(this.width, this.height, {
        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
        // Half float, not 8-bit.  An 8-bit alpha can only fall in steps of
        // 1/255, so however long the trail was asked to last it was gone after
        // 255 frames: 4.25 s on a 60 Hz screen, 1.8 s on a 144 Hz one, and the
        // trail-length setting above that did nothing at all.
        format: THREE.RGBAFormat, type: THREE.HalfFloatType,
        depthBuffer: false, stencilBuffer: false,
      }));
    this.material.uniforms.resolution.value.set(this.width, this.height);
    this.reset();
  }

  /**
   * The slice quad's world size.  Passed in rather than read off the geometry:
   * BufferGeometry.clone() does not carry PlaneGeometry's `parameters`, so a
   * cloned quad reports 1x1 and every projection would land in the wrong place.
   */
  setPlane(worldWidth, worldHeight) {
    this.planeSize.set(worldWidth, worldHeight);
  }

  /** `field` is {u, v, value, solid, width, height} over the wind-frame grid. */
  setField(field, metresPerCell) {
    this.field = field;
    this.metresPerCell = metresPerCell;
    this.ready = 1;
    this.readyX = Infinity;
    this.reset();
  }

  /**
   * Report how far a progressive fill has got: `fraction` of the patches done,
   * reaching `edge` cells downwind.  Until it completes, the count follows the
   * fraction so the density in the finished part stays what it will be at the
   * end, instead of every particle crowding into the first few columns.
   */
  setProgress(fraction, edge) {
    this.ready = Math.max(0, Math.min(1, fraction));
    this.readyX = Math.max(this.readyX === Infinity ? 0 : this.readyX, edge);
  }

  /**
   * Swap in a field without wiping the streaks.  Used when a prediction
   * finishes: the flow the particles have been running on was merged in region
   * by region as it arrived, and this replaces it with the exact final one --
   * clearing at that moment would blank the picture just as it completes.
   */
  replaceField(field) {
    this.field = field;
    this.ready = 1;
    this.readyX = Infinity;
  }

  /**
   * Colour the streaks by the displayed field instead of white.
   *
   * `palette` is the same 256-entry ramp the slice shader samples, generated
   * from the same colour() the legend draws with, so the three can never
   * disagree.  `range` is in reference-speed units, exactly as the slice uses
   * it, and is null to go back to plain white.
   */
  /** The settings block in force this frame. */
  get p() { return this.settings[this.mode] || this.settings.white; }

  /**
   * Which set of settings is live.  "off" is ignored: there is nothing to draw,
   * and keeping the last mode means the panel still edits something sensible.
   */
  setMode(mode) {
    if (mode !== "white" && mode !== "tinted") return;
    if (mode === this.mode) return;
    this.mode = mode;
    this.applyUniforms();
  }

  /** Change one setting of one mode, live. */
  setParam(mode, key, value) {
    const block = this.settings[mode];
    if (!block || !(key in block) || !Number.isFinite(value)) return;
    block[key] = value;
    if (mode === this.mode) this.applyUniforms();
  }

  /** Push the settings that live in uniforms; the rest are read each frame. */
  applyUniforms() {
    const p = this.p;
    this.fade.uniforms.cutoff.value = p.trailCutoff;
    this.lines.material.uniforms.halfWidth.value = p.lineWidth * 0.5;
    this.lines.material.uniforms.opacity.value = p.lineOpacity;
    this.material.uniforms.strength.value = p.strength;
    this.material.uniforms.solid.value = p.solidFraction;
  }

  setTinting(palette, range) {
    this.palette = palette || null;
    this.tintRange = range || null;
  }

  /** RGB in 0-1 for a field value, through the ramp the slice is using. */
  tintFor(value, out) {
    if (!this.palette || !this.tintRange) {
      out[0] = 1; out[1] = 1; out[2] = 1;
      return out;
    }
    const [low, high] = this.tintRange;
    const t = Math.min(1, Math.max(0, (value - low) / Math.max(1e-6, high - low)));
    const index = Math.min(255, Math.round(t * 255)) * 4;
    out[0] = this.palette[index] / 255;
    out[1] = this.palette[index + 1] / 255;
    out[2] = this.palette[index + 2] / 255;
    return out;
  }

  reset() {
    this.particles.length = 0;
    this.viewKey = "";
    if (!this.targets) return;
    const renderer = this.renderer;
    const previous = renderer.getRenderTarget();
    const colour = new THREE.Color();
    const alpha = renderer.getClearAlpha();
    renderer.getClearColor(colour);
    renderer.setClearColor(0x000000, 0);
    for (const target of this.targets) {
      renderer.setRenderTarget(target);
      renderer.clear(true, false, false);
    }
    renderer.setClearColor(colour, alpha);
    renderer.setRenderTarget(previous);
  }

  /** Bilinear velocity in cells per second, or null where there is no flow. */
  sample(x, y, out) {
    const f = this.field;
    if (!f || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (x < 0.5 || y < 0.5 || x > f.width - 0.5 || y > f.height - 0.5) {
      return null;
    }
    const nx = Math.min(f.width - 1, Math.max(0, Math.round(x - 0.5)));
    const ny = Math.min(f.height - 1, Math.max(0, Math.round(y - 0.5)));
    if (f.solid[ny * f.width + nx] !== 0) return null;
    const gx = Math.min(f.width - 1, Math.max(0, x - 0.5));
    const gy = Math.min(f.height - 1, Math.max(0, y - 0.5));
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const x1 = Math.min(x0 + 1, f.width - 1);
    const y1 = Math.min(y0 + 1, f.height - 1);
    const tx = gx - x0;
    const ty = gy - y0;
    const blend = (channel) => {
      const a = channel[y0 * f.width + x0];
      const b = channel[y0 * f.width + x1];
      const c = channel[y1 * f.width + x0];
      const d = channel[y1 * f.width + x1];
      return (a * (1 - tx) + b * tx) * (1 - ty)
           + (c * (1 - tx) + d * tx) * ty;
    };
    out[0] = blend(f.u) / this.metresPerCell;
    out[1] = blend(f.v) / this.metresPerCell;
    this.lastValue = f.value ? blend(f.value) : 0;
    return Math.hypot(out[0], out[1]) > 1e-3 ? out : null;
  }

  /** The part of the grid the camera can actually see, in cells. */
  visibleBounds(camera, mesh) {
    const f = this.field;
    const full = { x0: 0.5, y0: 0.5, x1: f.width - 0.5, y1: f.height - 0.5 };
    const normal = this.scratch.set(0, 0, 1).applyQuaternion(mesh.quaternion);
    this.plane.setFromNormalAndCoplanarPoint(normal.clone(), mesh.position);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [nx, ny] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const near = new THREE.Vector3(nx, ny, -1).unproject(camera);
      const far = new THREE.Vector3(nx, ny, 1).unproject(camera);
      this.ray.set(near, far.sub(near).normalize());
      const hit = this.ray.intersectPlane(this.plane, new THREE.Vector3());
      if (!hit) return full;                     // plane edge-on or behind us
      const local = mesh.worldToLocal(hit);
      const cellX = (local.x / this.planeSize.x + 0.5) * f.width;
      const cellY = (local.y / this.planeSize.y + 0.5) * f.height;
      x0 = Math.min(x0, cellX); x1 = Math.max(x1, cellX);
      y0 = Math.min(y0, cellY); y1 = Math.max(y1, cellY);
    }
    return {
      x0: Math.max(full.x0, x0), x1: Math.min(full.x1, x1),
      y0: Math.max(full.y0, y0), y1: Math.min(full.y1, y1),
    };
  }

  respawn(particle, bounds) {
    particle.age = 0;
    particle.life = 0;
    particle.x = NaN;
    particle.y = NaN;
    // A degenerate visible box would leave every particle unplaced for ever,
    // which looks exactly like the flow being switched off.  Fall back to the
    // whole domain rather than quietly drawing nothing.
    if (bounds.x1 <= bounds.x0 || bounds.y1 <= bounds.y0) {
      bounds = { x0: 0.5, y0: 0.5,
                 x1: Math.min(this.readyX, this.field.width - 0.5),
                 y1: this.field.height - 0.5 };
      this.degenerate = true;
    }
    const velocity = [0, 0];
    for (let attempt = 0; attempt < 40; attempt++) {
      const x = bounds.x0 + Math.random() * (bounds.x1 - bounds.x0);
      const y = bounds.y0 + Math.random() * (bounds.y1 - bounds.y0);
      if (this.sample(x, y, velocity)) {
        particle.x = x;
        particle.y = y;
        // A particle drawing a streak longer than it lives stops growing it
        // halfway: the streak is at most `life` seconds of path.  So lifetime
        // is tied to the trail length instead of being set on its own, where
        // raising the trail without also raising two lifetime sliders just
        // produced the same short streaks.
        const [low, high] = LIFE_PER_TRAIL;
        particle.life = this.p.trailSeconds * (low + Math.random() * (high - low));
        return true;
      }
    }
    return false;
  }

  /** Grid cell to pixels, through the same transform the plane is drawn with. */
  toPixels(x, y, mesh, out) {
    const f = this.field;
    const local = this.scratch.set(
      (x / f.width - 0.5) * this.planeSize.x,
      (y / f.height - 0.5) * this.planeSize.y, 0);
    local.applyMatrix4(this.matrix);
    out[0] = (local.x * 0.5 + 0.5) * this.width;
    out[1] = (0.5 - local.y * 0.5) * this.height;
    return out;
  }

  track(x, y) {
    const e = this.extent;
    if (x < e.x0) e.x0 = x;
    if (x > e.x1) e.x1 = x;
    if (y < e.y0) e.y0 = y;
    if (y > e.y1) e.y1 = y;
  }

  addSegment(offset, start, end, tint) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const length = Math.hypot(dx, dy);
    if (length === 0) return offset;
    const half = this.p.lineWidth * 0.5;
    const ox = -dy * half / length;
    const oy = dx * half / length;
    // Grow the quad past both endpoints so it can hold the round caps the
    // fragment shader draws inside it.
    const ex = dx * half / length;
    const ey = dy * half / length;
    const x0 = start[0] - ex, y0 = start[1] - ey;
    const x1 = end[0] + ex, y1 = end[1] + ey;
    const sx = 2 / this.width;
    const sy = 2 / this.height;
    const ax = (x0 + ox) * sx - 1, ay = 1 - (y0 + oy) * sy;
    const bx = (x0 - ox) * sx - 1, by = 1 - (y0 - oy) * sy;
    const cx = (x1 + ox) * sx - 1, cy = 1 - (y1 + oy) * sy;
    const dxc = (x1 - ox) * sx - 1, dyc = 1 - (y1 - oy) * sy;
    this.vertices.set([ax, ay, bx, by, cx, cy, cx, cy, bx, by, dxc, dyc],
                      offset);
    const tintBase = offset * 3 / 2;      // six vertices, three floats each
    const endBase = offset * 2;           // six vertices, four floats each
    // gl_FragCoord counts from the bottom, the pixel coordinates here from the
    // top, so the endpoints go in flipped.
    const capsule = [start[0], this.height - start[1],
                     end[0], this.height - end[1]];
    for (let vertex = 0; vertex < 6; vertex++) {
      this.tints.set(tint, tintBase + vertex * 3);
      this.ends.set(capsule, endBase + vertex * 4);
    }
    this.track(ax, ay);
    this.track(cx, cy);
    return offset + 12;
  }

  /**
   * Advance and draw.  Returns false when nothing was drawn, so the caller can
   * tell whether the overlay has anything to show.
   */
  step(deltaSeconds, camera, slice) {
    if (!this.enabled || !this.field || !this.targets || !slice) return false;
    if (deltaSeconds > 0.1) { this.reset(); return false; }

    slice.updateWorldMatrix(true, false);
    this.matrix.multiplyMatrices(camera.projectionMatrix,
      camera.matrixWorldInverse).multiply(slice.matrixWorld);

    // Screen-space streaks are only valid for the view that drew them.
    const key = camera.matrixWorldInverse.elements.map(
      (value) => value.toFixed(4)).join(",");
    if (key !== this.viewKey) { this.reset(); this.viewKey = key; }

    const bounds = this.visibleBounds(camera, slice);
    // Spawning is confined to the part of the grid that has arrived; the fill
    // marches downwind, so that is everything up to readyX.
    const spawn = { x0: bounds.x0, y0: bounds.y0, y1: bounds.y1,
                    x1: Math.min(bounds.x1, this.readyX) };
    const seen = ((bounds.x1 - bounds.x0) * (bounds.y1 - bounds.y0))
      / (this.field.width * this.field.height);
    const p = this.p;
    const full = Math.min(p.maxParticles, PARTICLE_CAPACITY,
      Math.round(this.width * this.height / Math.max(1, p.pixelsPerParticle)));
    const target = Math.min(PARTICLE_CAPACITY, Math.round(
      Math.max(p.minParticles, full * Math.sqrt(Math.max(0, Math.min(1, seen))))
      * this.ready));
    while (this.particles.length < target) {
      const particle = {};
      this.respawn(particle, spawn);
      this.particles.push(particle);
    }
    if (this.particles.length > target) this.particles.length = target;

    this.extent = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
    const stepScale = this.p.timeScale * deltaSeconds;
    const velocity = [0, 0];
    const start = [0, 0];
    const end = [0, 0];
    const tint = [1, 1, 1];
    let offset = 0;
    for (const particle of this.particles) {
      if (particle.age >= particle.life) this.respawn(particle, spawn);
      if (!this.sample(particle.x, particle.y, velocity)) {
        this.respawn(particle, spawn);
        continue;                      // recycled: draw nothing this frame
      }
      const midX = particle.x + 0.5 * velocity[0] * stepScale;
      const midY = particle.y + 0.5 * velocity[1] * stepScale;
      if (!this.sample(midX, midY, velocity)) {
        this.respawn(particle, spawn);
        continue;
      }
      const nextX = particle.x + velocity[0] * stepScale;
      const nextY = particle.y + velocity[1] * stepScale;
      // Leaving the visible box retires the particle, but a particle that
      // simply runs past the fill front has not left anything -- the sample
      // there fails and recycles it, which is the same outcome by the right
      // reason.
      if (nextX < bounds.x0 || nextX > bounds.x1
          || nextY < bounds.y0 || nextY > bounds.y1
          || !this.sample(nextX, nextY, velocity)) {
        this.respawn(particle, spawn);
        continue;
      }
      // lastValue was left by the sample at the landing point, so the segment
      // is coloured by the field where the particle arrives.
      this.tintFor(this.lastValue, tint);
      offset = this.addSegment(offset,
        this.toPixels(particle.x, particle.y, slice, start),
        this.toPixels(nextX, nextY, slice, end), tint);
      particle.x = nextX;
      particle.y = nextY;
      particle.age += deltaSeconds;
    }
    this.stats = {
      target,
      drawn: offset / 12,
      placed: this.particles.filter((p) => Number.isFinite(p.x)).length,
      bounds: [bounds.x0, bounds.y0, bounds.x1, bounds.y1]
        .map((value) => value.toFixed(0)).join(".."),
    };
    this.draw(deltaSeconds, offset);
    return true;
  }

  draw(deltaSeconds, floatCount) {
    const renderer = this.renderer;
    const previousTarget = renderer.getRenderTarget();
    const autoClear = renderer.autoClear;
    renderer.autoClear = false;
    this.fade.uniforms.previous.value = this.targets[0].texture;
    // The floor only guarantees a streak eventually reaches the cutoff; with a
    // half-float buffer it no longer caps how long a streak can be.
    this.fade.uniforms.drop.value =
      Math.max(1e-4, deltaSeconds / Math.max(0.01, this.p.trailSeconds));
    this.quad.material = this.fade;
    renderer.setRenderTarget(this.targets[1]);
    renderer.render(this.quadScene, this.camera);

    if (floatCount > 0) {
      this.lineGeometry.setDrawRange(0, floatCount / 2);
      this.lineGeometry.attributes.segment.needsUpdate = true;
      this.lineGeometry.attributes.tint.needsUpdate = true;
      this.lineGeometry.attributes.ends.needsUpdate = true;
      renderer.render(this.lineScene, this.camera);
    }
    this.targets.reverse();
    this.material.uniforms.trail.value = this.targets[0].texture;
    renderer.autoClear = autoClear;
    renderer.setRenderTarget(previousTarget);
  }

  dispose() {
    for (const target of this.targets || []) target.dispose();
    this.lineGeometry.dispose();
    this.lines.material.dispose();
    this.material.dispose();
    this.fade.dispose();
  }
}
