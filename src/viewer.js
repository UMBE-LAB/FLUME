// Three.js scene: extruded buildings, the oblique prediction plane, the
// particle layer and a deliberately constrained camera.
//
// Camera policy (WEB_DEMO_HANDOFF section 2): the horizontal viewing direction
// is fixed so that city orientation can never be confused with wind direction.
// Only pitch, pan and zoom are exposed.

import * as THREE from "../vendor/three.module.js";
import { HORIZONTAL_SPACING_M, VERTICAL_SPACING_M } from "./city.js";

const PARTICLE_COUNT = 6000;

// The slice is drawn from a scalar field plus a colour ramp, not from a
// pre-coloured image, so changing the colour range is two uniforms rather than
// re-colouring every cell on the CPU and re-uploading the texture.
//
// ShaderMaterial does not get the clipping chunks for free the way the built-in
// materials do: they have to be included by hand and `clipping` set to true, or
// the slice will overhang the city footprint.
const SLICE_VERTEX = `
#include <clipping_planes_pars_vertex>
varying vec2 vUv;
void main() {
  vUv = uv;
  vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;

const SLICE_FRAGMENT = `
#include <clipping_planes_pars_fragment>
uniform sampler2D field;      // [value, alpha, unused, unused]
uniform sampler2D palette;    // 256 x 1 colour ramp
uniform vec2 range;           // low, high of the current colour scale
varying vec2 vUv;
void main() {
  #include <clipping_planes_fragment>
  vec4 texel = texture2D( field, vUv );
  // No hard cut: the alpha ramps over one cell where the field meets a building
  // or the city edge, and letting it ramp is what keeps that edge smooth
  // instead of stepping from cell to cell.
  if ( texel.g <= 0.002 ) discard;
  float t = clamp( ( texel.r - range.x ) / max( range.y - range.x, 1e-6 ),
                   0.0, 1.0 );
  gl_FragColor = vec4( texture2D( palette, vec2( t, 0.5 ) ).rgb, texel.g );
  #include <colorspace_fragment>
}`;

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.localClippingEnabled = true;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0f1115);

    this.camera = new THREE.PerspectiveCamera(38, 1, 1, 20000);
    this.camera.up.set(0, 0, 1);
    this.target = new THREE.Vector3();
    this.distance = 2200;
    this.pitch = 42;
    this.panOffset = new THREE.Vector2();

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(-1, -1.2, 2);
    this.scene.add(key);

    this.buildings = null;
    this.ground = null;
    this.slice = null;
    this.streaks = null;
    this.onResize = null;
    // Fired whenever the camera or the canvas changes, so the scale bar can
    // restate itself.  Deliberately not fired from render(): render runs every
    // animation frame while particles are moving, and the scale only changes
    // when the camera does.
    this.onCamera = null;
    this.arrow = null;
    this.clipPlanes = [];
    this.city = null;
    this.frame = null;          // the particle animation loop, when running
    this.pendingDraw = null;    // a coalesced one-off draw, see render()
    this.material = null;
    this.attachInteraction();
    this.resize();
    window.addEventListener("resize", () => this.resize());
    // The canvas may still be unlaid-out at construction; observe it so the
    // first real size is picked up instead of staying 1x1.
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  attachInteraction() {
    const canvas = this.canvas;
    let button = null;          // 0 = left, pans; 2 = right, changes pitch
    let last = null;
    // Without this the browser's own menu opens on the first right-drag.
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointerdown", (event) => {
      button = event.button;
      last = [event.clientX, event.clientY];
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (button === null) return;
      const moveX = event.clientX - last[0];
      const moveY = event.clientY - last[1];
      last = [event.clientX, event.clientY];
      if (button === 2) {
        // Drag down to look from further above.  Azimuth stays locked (see
        // the camera policy at the top of this file), so only pitch moves.
        this.setPitch(this.pitch + moveY * 0.25);
        return;
      }
      const scale = this.distance * 0.0016;
      // Pan in the fixed horizontal viewing frame.
      this.target.x -= moveX * scale;
      this.target.y += moveY * scale
        * Math.max(0.35, Math.cos(this.pitch * Math.PI / 180));
      this.updateCamera();
    });
    const stop = (event) => {
      button = null;
      if (canvas.hasPointerCapture?.(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.distance = Math.min(12000, Math.max(150,
        this.distance * Math.exp(event.deltaY * 0.0012)));
      this.updateCamera();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => this.resetCamera());
  }

  /** Everything a black viewport could be blamed on, in one string. */
  diagnostics() {
    const size = this.renderer.getSize(new THREE.Vector2());
    return `canvas ${this.canvas.clientWidth}x${this.canvas.clientHeight} ` +
      `drawing ${size.x}x${size.y} · objects ${this.scene.children.length} ` +
      `· camera (${this.camera.position.toArray().map((v) => v.toFixed(0))}) ` +
      `-> (${this.target.toArray().map((v) => v.toFixed(0))}) d=${this.distance.toFixed(0)}`;
  }

  get buildingCount() { return this.buildings ? this.buildings.count : 0; }

  resize() {
    const width = this.canvas.clientWidth || 1;
    const height = this.canvas.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    const buffer = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.onResize?.(buffer.x, buffer.y);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.onCamera?.();
    this.render();
  }

  setPitch(degrees) {
    this.pitch = Math.min(85, Math.max(8, degrees));
    this.updateCamera();
  }

  resetCamera() {
    if (!this.city) return;
    const spanX = this.city.nx * HORIZONTAL_SPACING_M;
    const spanY = this.city.ny * HORIZONTAL_SPACING_M;
    this.target.set(spanX / 2, spanY / 2, 0);
    this.distance = Math.max(spanX, spanY) * 1.9;
    this.updateCamera();
  }

  updateCamera() {
    const pitch = this.pitch * Math.PI / 180;
    // Fixed azimuth: the observer always looks from -Y towards +Y.
    this.camera.position.set(
      this.target.x,
      this.target.y - this.distance * Math.cos(pitch),
      this.target.z + this.distance * Math.sin(pitch));
    this.camera.lookAt(this.target);
    this.onCamera?.();
    this.render();
  }

  /**
   * Metres per CSS pixel measured horizontally, at the depth of the camera
   * target.
   *
   * The azimuth is fixed looking along +Y, so world X is exactly screen X and
   * exactly horizontal: a bar drawn across the screen is not foreshortened by
   * the pitch, and the only approximation left is that objects nearer than the
   * target are larger.  That makes the figure exact at the centre of the view
   * rather than merely indicative, which is the usual standard for a scale bar
   * on a perspective view.
   */
  metresPerPixel() {
    const height = this.canvas.clientHeight || 1;
    const fov = this.camera.fov * Math.PI / 180;
    return 2 * this.distance * Math.tan(fov / 2) / height;
  }

  /** Rebuild the extruded city and the ground plane. */
  setCity(city) {
    this.city = city;
    for (const object of [this.buildings, this.ground]) {
      if (object) {
        this.scene.remove(object);
        object.geometry.dispose();
        object.material.dispose();
      }
    }
    const spanX = city.nx * HORIZONTAL_SPACING_M;
    const spanY = city.ny * HORIZONTAL_SPACING_M;

    this.ground = new THREE.Mesh(
      new THREE.PlaneGeometry(spanX, spanY),
      new THREE.MeshStandardMaterial({ color: 0x23262d, roughness: 1 }));
    this.ground.position.set(spanX / 2, spanY / 2, -0.5);
    this.scene.add(this.ground);

    let built = 0;
    for (let i = 0; i < city.layers.length; i++) if (city.layers[i] > 0) built++;
    const geometry = new THREE.BoxGeometry(
      HORIZONTAL_SPACING_M, HORIZONTAL_SPACING_M, 1);
    const material = new THREE.MeshStandardMaterial({
      color: 0x9aa3ad, roughness: 0.85, metalness: 0.0,
    });
    const mesh = new THREE.InstancedMesh(geometry, material, Math.max(1, built));
    const matrix = new THREE.Matrix4();
    let index = 0;
    for (let y = 0; y < city.ny; y++) {
      for (let x = 0; x < city.nx; x++) {
        const layers = city.layers[y * city.nx + x];
        if (layers <= 0) continue;
        const height = layers * VERTICAL_SPACING_M;
        matrix.makeScale(1, 1, height);
        matrix.setPosition(
          (x + 0.5) * HORIZONTAL_SPACING_M,
          (y + 0.5) * HORIZONTAL_SPACING_M,
          height / 2);
        mesh.setMatrixAt(index++, matrix);
      }
    }
    mesh.count = index;
    mesh.instanceMatrix.needsUpdate = true;
    this.buildings = mesh;
    this.scene.add(mesh);

    // The oblique slice is clipped to the city footprint.
    this.clipPlanes = [
      new THREE.Plane(new THREE.Vector3(1, 0, 0), 0),
      new THREE.Plane(new THREE.Vector3(-1, 0, 0), spanX),
      new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
      new THREE.Plane(new THREE.Vector3(0, -1, 0), spanY),
    ];
    if (this.material) this.material.clippingPlanes = this.clipPlanes;
    this.resetCamera();
  }

  /**
   * Place the prediction plane.  The texture lives on the wind-frame grid, so
   * the quad is rotated by theta about the domain centre instead of resampling.
   */
  /** The one material for the slice; its uniforms are what animation drives. */
  sliceMaterial() {
    if (!this.material) {
      this.material = new THREE.ShaderMaterial({
        vertexShader: SLICE_VERTEX,
        fragmentShader: SLICE_FRAGMENT,
        uniforms: {
          field: { value: null },
          palette: { value: null },
          range: { value: new THREE.Vector2(0, 1) },
        },
        transparent: true, depthWrite: false, side: THREE.DoubleSide,
        clipping: true, clippingPlanes: this.clipPlanes,
      });
    }
    this.material.clippingPlanes = this.clipPlanes;
    return this.material;
  }

  /** Swap the colour ramp; only the variable changing needs this. */
  setPalette(texture) {
    this.sliceMaterial().uniforms.palette.value = texture;
    this.render();
  }

  /** The whole point of the shader path: a scale change costs one uniform. */
  setRange(low, high) {
    this.sliceMaterial().uniforms.range.value.set(low, high);
    this.render();
  }

  setSlice(texture, layout, heightMetres) {
    if (this.slice) {
      this.scene.remove(this.slice);
      this.slice.geometry.dispose();
    }
    const width = layout.width * HORIZONTAL_SPACING_M;
    const height = layout.height * HORIZONTAL_SPACING_M;
    const material = this.sliceMaterial();
    material.uniforms.field.value = texture;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height), material);
    const centreX = (this.city.nx - 1) / 2;
    const centreY = (this.city.ny - 1) / 2;
    // Centre of the wind-frame grid, expressed in city cells then rotated.
    const sx = layout.originX + layout.width / 2 - 0.5;
    const sy = layout.originY + layout.height / 2 - 0.5;
    const dx = sx - centreX;
    const dy = sy - centreY;
    const cos = Math.cos(layout.theta);
    const sin = Math.sin(layout.theta);
    const ix = centreX + dx * cos - dy * sin;
    const iy = centreY + dx * sin + dy * cos;
    mesh.position.set(
      (ix + 0.5) * HORIZONTAL_SPACING_M,
      (iy + 0.5) * HORIZONTAL_SPACING_M,
      heightMetres);
    mesh.rotation.z = layout.theta;
    mesh.renderOrder = 2;
    this.slice = mesh;
    this.scene.add(mesh);
    if (this.streaks) this.setStreaks(this.streaks);
    // Give the field texture GPU storage explicitly, before anything copies
    // into it.  The progressive fill writes through copyTextureToTexture, which
    // is a texSubImage2D into an existing allocation; until now the only thing
    // establishing that allocation was the side effect of a render happening to
    // draw this material first, which is not something to depend on.
    this.renderer.initTexture(texture);
    this.draw();
  }

  /**
   * Upload one rectangle into an already-resident texture.  This is the whole
   * point of the progressive path: the cost of a refresh is the size of the
   * rectangle, not the size of the field, so refreshing often is free.
   */
  uploadRegion(texels, width, height, x, y, destination) {
    const source = new THREE.DataTexture(
      texels, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
    source.needsUpdate = true;
    this.renderer.copyTextureToTexture(
      source, destination, null, new THREE.Vector2(x, y));
    source.dispose();
    this.render();
  }

  /**
   * The streak ribbons are their own mesh, given the slice's transform so they
   * sit on the plane.  Being geometry, they are depth-tested against the
   * buildings and stay sharp however far the camera zooms in.
   */
  setStreaks(mesh) {
    if (this.streaks && this.streaks !== mesh) this.scene.remove(this.streaks);
    this.streaks = mesh || null;
    if (!mesh || !this.slice) return;
    mesh.material.clippingPlanes = this.clipPlanes;
    mesh.renderOrder = 3;
    // Same quad as the slice, so the trail is depth-tested and clipped exactly
    // as the field beneath it is.
    mesh.geometry.dispose();
    mesh.geometry = this.slice.geometry.clone();
    mesh.position.copy(this.slice.position);
    mesh.rotation.copy(this.slice.rotation);
    if (!mesh.parent) this.scene.add(mesh);
    this.render();
  }

  /** Hide the field itself, for the mode where the particles carry it. */
  showSlice(on) {
    if (this.slice) this.slice.visible = on;
    this.render();
  }

  showStreaks(on) {
    if (this.streaks) this.streaks.visible = on;
    this.render();
  }

  setInflow(theta) {
    if (this.arrow) this.scene.remove(this.arrow);
    if (!this.city) return;
    const spanX = this.city.nx * HORIZONTAL_SPACING_M;
    const spanY = this.city.ny * HORIZONTAL_SPACING_M;
    const length = Math.max(spanX, spanY) * 0.22;
    const direction = new THREE.Vector3(Math.cos(theta), Math.sin(theta), 0);
    const origin = new THREE.Vector3(spanX / 2, spanY / 2, 0)
      .addScaledVector(direction, -Math.max(spanX, spanY) * 0.72);
    this.arrow = new THREE.ArrowHelper(
      direction, origin, length, 0x64d2ff, length * 0.3, length * 0.16);
    this.scene.add(this.arrow);
    this.render();
  }

  /**
   * Ask for a redraw on the next animation frame.
   *
   * Everything on this page draws on demand: with the particles off -- which
   * is how the page starts -- there is no animation loop at all, and every
   * render() call arrives from a promise continuation or an event handler.
   * Drawing straight from a microtask means drawing outside the frame the
   * compositor is about to present, and on the very first load, before the
   * canvas has ever been composited, the result is a blank viewport that stays
   * blank until something else causes a frame.  Any interaction did: a pointer
   * event or a slider is followed by a paint, which is why the scene appeared
   * the moment anything was touched.
   *
   * Deferring to requestAnimationFrame puts every draw inside a frame, and
   * coalesces them as well: a progressive fill calls this once per uploaded
   * rectangle and the colour glide once per frame, and now at most one draw
   * comes out of each frame either way.
   */
  render() {
    if (this.frame !== null) return;        // the animation loop already draws
    if (this.pendingDraw !== null) return;
    this.pendingDraw = requestAnimationFrame(() => {
      this.pendingDraw = null;
      this.draw();
    });
  }

  /**
   * Draw now, without waiting for a frame.
   *
   * Only for the two places that need the GPU-side state a render establishes
   * before the very next statement runs: uploading a texture for the first
   * time, which is what makes copyTextureToTexture legal against it.
   */
  draw() {
    if (this.pendingDraw !== null) {
      cancelAnimationFrame(this.pendingDraw);
      this.pendingDraw = null;
    }
    this.renderer.render(this.scene, this.camera);
  }

  /** Continuous rendering only while particles are animating. */
  animate(step) {
    if (this.frame) cancelAnimationFrame(this.frame);
    const loop = () => {
      step();
      this.draw();
      this.frame = requestAnimationFrame(loop);
    };
    this.frame = requestAnimationFrame(loop);
  }

  stopAnimation() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = null;
    // Leave the last frame on screen, but make sure one more is presented:
    // the loop may have been cancelled mid-frame.
    this.render();
  }
}
