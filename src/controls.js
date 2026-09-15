// Left-column controls and the in-viewport sliders.
//
// Every physical control pairs a direct-manipulation widget with a numeric
// field, and the two stay synchronized: the numbers are the exact input path.

import { VARIABLES } from "./field.js";
import { thumbnailOf, thumbnailFrom,
         HORIZONTAL_SPACING_M } from "./city.js";

// Backing resolution of one city tile; the CSS size follows the viewport.
const TILE_PIXELS = 96;
// The widest a scale bar is allowed to be drawn, in CSS pixels.
const SCALE_MAX_PX = 170;

/** Slider plus number input driving one scalar. */
function linkNumeric(range, number, onChange, format = (v) => v) {
  let value = parseFloat(range.value);
  const push = (next, source) => {
    if (!Number.isFinite(next)) return;
    value = Math.min(parseFloat(range.max), Math.max(parseFloat(range.min), next));
    if (source !== "range") range.value = String(value);
    if (source !== "number") number.value = String(format(value));
    onChange(value);
  };
  range.addEventListener("input", () => push(parseFloat(range.value), "range"));
  number.addEventListener("change",
    () => push(parseFloat(number.value), "number"));
  return {
    get value() { return value; },
    set(next) { push(next, null); },
    silent(next) {
      value = next;
      range.value = String(next);
      number.value = String(format(next));
    },
  };
}

/** Large rotary control for wind direction; drag anywhere on the dial face. */
class Dial {
  constructor(canvas, numberInput, onChange) {
    this.canvas = canvas;
    this.number = numberInput;
    this.onChange = onChange;
    this.degrees = 0;
    this.dragging = false;

    const pointerAngle = (event) => {
      const rect = canvas.getBoundingClientRect();
      const dx = event.clientX - rect.left - rect.width / 2;
      const dy = event.clientY - rect.top - rect.height / 2;
      // Screen y grows downwards; the physical frame is counter-clockwise.
      return Math.round(Math.atan2(-dy, dx) * 180 / Math.PI);
    };
    canvas.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      canvas.setPointerCapture(event.pointerId);
      this.set(pointerAngle(event));
    });
    canvas.addEventListener("pointermove", (event) => {
      if (this.dragging) this.set(pointerAngle(event));
    });
    const stop = (event) => {
      this.dragging = false;
      if (canvas.hasPointerCapture?.(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    numberInput.addEventListener("change", () => {
      this.set(parseFloat(numberInput.value));
    });
    this.draw();
  }

  set(degrees) {
    if (!Number.isFinite(degrees)) return;
    let normalized = degrees % 360;
    if (normalized > 180) normalized -= 360;
    if (normalized <= -180) normalized += 360;
    this.degrees = normalized;
    this.number.value = String(Math.round(normalized));
    this.draw();
    this.onChange(normalized * Math.PI / 180);
  }

  silent(degrees) {
    this.degrees = degrees;
    this.number.value = String(Math.round(degrees));
    this.draw();
  }

  draw() {
    const c = this.canvas.getContext("2d");
    const size = this.canvas.width;
    const centre = size / 2;
    const radius = centre - 16;
    c.clearRect(0, 0, size, size);

    c.strokeStyle = "#262c36";
    c.lineWidth = 2;
    c.beginPath();
    c.arc(centre, centre, radius, 0, Math.PI * 2);
    c.stroke();

    c.fillStyle = "#8b95a3";
    c.font = "11px system-ui, sans-serif";
    c.textAlign = "center";
    c.textBaseline = "middle";
    for (const [label, angle] of
         [["0°", 0], ["90°", 90], ["180°", 180], ["270°", 270]]) {
      const radians = angle * Math.PI / 180;
      c.fillText(label,
        centre + Math.cos(radians) * (radius + 9),
        centre - Math.sin(radians) * (radius + 9));
      c.beginPath();
      c.moveTo(centre + Math.cos(radians) * (radius - 7),
               centre - Math.sin(radians) * (radius - 7));
      c.lineTo(centre + Math.cos(radians) * radius,
               centre - Math.sin(radians) * radius);
      c.stroke();
    }

    const radians = this.degrees * Math.PI / 180;
    const tipX = centre + Math.cos(radians) * (radius - 14);
    const tipY = centre - Math.sin(radians) * (radius - 14);
    c.strokeStyle = "#64d2ff";
    c.lineWidth = 3;
    c.beginPath();
    c.moveTo(centre - Math.cos(radians) * (radius - 24),
             centre + Math.sin(radians) * (radius - 24));
    c.lineTo(tipX, tipY);
    c.stroke();
    c.fillStyle = "#64d2ff";
    c.beginPath();
    c.arc(tipX, tipY, 6, 0, Math.PI * 2);
    c.fill();
    c.beginPath();
    c.arc(centre, centre, 3.5, 0, Math.PI * 2);
    c.fill();
  }
}

export function buildControls(handlers) {
  const $ = (id) => document.getElementById(id);

  const speed = linkNumeric($("speed-range"), $("speed-number"),
    handlers.onSpeed, (v) => v.toFixed(1));

  const dial = new Dial($("dial"), $("direction-number"), handlers.onDirection);

  const heightRange = $("height-range");
  const heightNumber = $("height-number");
  const applyHeight = (layer, source) => {
    const clamped = Math.min(parseInt(heightRange.max, 10),
      Math.max(0, Math.round(layer)));
    if (source !== "range") heightRange.value = String(clamped);
    heightNumber.value = ((clamped + 0.5) * 1.5).toFixed(1);
    return clamped;
  };
  // Dragging only moves the readout; the recompute waits for the release.
  // Committing on "input" started an MDDF trace on the first pixel of the drag
  // and then chased the pointer the whole way, which is the opposite of what
  // this group is meant to say about its cost.
  heightRange.addEventListener("input",
    () => applyHeight(parseInt(heightRange.value, 10), "range"));
  heightRange.addEventListener("change", () => {
    handlers.onHeight(applyHeight(parseInt(heightRange.value, 10), "range"));
  });
  heightNumber.addEventListener("change", () => {
    const metres = parseFloat(heightNumber.value);
    if (!Number.isFinite(metres)) return;
    handlers.onHeight(applyHeight(metres / 1.5 - 0.5, "number"));
  });

  const variables = $("variables");
  const buttons = VARIABLES.map((variable) => {
    const button = document.createElement("button");
    button.textContent = variable.label;
    button.type = "button";
    button.title = variable.legend;
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      buttons.forEach((other) =>
        other.setAttribute("aria-pressed", String(other === button)));
      handlers.onVariable(variable.id);
    });
    variables.append(button);
    return button;
  });

  $("particles").addEventListener("change",
    (event) => handlers.onParticles(event.target.value));

  $("city-upload").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) handlers.onUpload(file);
  });

  const tiles = $("city-tiles");
  const chosen = new Map();          // city id -> its tile button
  let uploadTile = null;
  const markCity = (id) => {
    for (const [key, button] of chosen) {
      button.setAttribute("aria-pressed", String(key === id));
    }
  };


  const scaleBar = document.querySelector("#scalebar .bar");
  const scaleLabel = $("scale-label");

  return {
    speed, dial,
    selectVariable(id) {
      const index = VARIABLES.findIndex((entry) => entry.id === id);
      buttons.forEach((button, position) =>
        button.setAttribute("aria-pressed", String(position === index)));
    },
    /**
     * One square tile per city.  The thumbnails are decoded after the picker
     * is on screen, so the first city can start predicting while they arrive;
     * a tile that fails to decode simply stays blank rather than taking the
     * page down.
     */
    setCities(manifest, selected) {
      tiles.replaceChildren();
      chosen.clear();
      uploadTile = null;
      for (const city of manifest.cities) {
        const button = document.createElement("button");
        button.type = "button";
        const [ny, nx] = city.shape;
        const kmX = nx * HORIZONTAL_SPACING_M / 1000;
        const kmY = ny * HORIZONTAL_SPACING_M / 1000;
        button.title = `${city.label} · ${kmX.toFixed(1)} × ${kmY.toFixed(1)} km`;
        button.setAttribute("aria-pressed", String(city.id === selected));
        button.addEventListener("click", () => {
          markCity(city.id);
          handlers.onCity(city.id);
        });
        tiles.append(button);
        chosen.set(city.id, button);
        thumbnailOf(manifest, city.id, TILE_PIXELS)
          .then((canvas) => button.replaceChildren(canvas))
          .catch(() => {});
      }
    },
    /** Add or replace the tile standing for the uploaded map, and select it. */
    async showUploaded(file) {
      if (!uploadTile) {
        uploadTile = document.createElement("button");
        uploadTile.type = "button";
        uploadTile.addEventListener("click", () => markCity("__upload__"));
        tiles.append(uploadTile);
        chosen.set("__upload__", uploadTile);
        // One more column rather than a second row: a second row would add
        // about 44 px to a panel that is sized not to scroll.
        tiles.style.gridTemplateColumns = `repeat(${chosen.size}, 1fr)`;
      }
      uploadTile.title = file.name;
      markCity("__upload__");
      try {
        uploadTile.replaceChildren(await thumbnailFrom(file, TILE_PIXELS));
      } catch { /* a blank tile is still selectable */ }
    },
    selectParticles(mode) { $("particles").value = mode; },
    setHeightBounds(maximumLayer) {
      heightRange.max = String(maximumLayer);
      heightNumber.max = ((maximumLayer + 0.5) * 1.5).toFixed(1);
    },
    setHeight(layer) { return applyHeight(layer, null); },
    setCityInfo(text) { $("city-info").textContent = text; },
    setStatus(text) { $("status").textContent = text; },
    setBusy(busy) { $("busy").hidden = !busy; },
    /**
     * Restate the scale bar for the current camera.
     *
     * The drawn length is chosen from 1/2/5 x 10^n so the number under the bar
     * is one a reader can hold in their head; the bar is then whatever pixel
     * length that number works out to, rather than a fixed bar carrying an
     * awkward number.
     */
    setScale(metresPerPixel) {
      if (!(metresPerPixel > 0)) return;
      let metres = 1;
      for (let exponent = 0; exponent <= 6; exponent++) {
        for (const step of [1, 2, 5]) {
          const candidate = step * 10 ** exponent;
          if (candidate / metresPerPixel <= SCALE_MAX_PX) metres = candidate;
        }
      }
      scaleBar.style.width = `${(metres / metresPerPixel).toFixed(1)}px`;
      scaleLabel.textContent = metres >= 1000
        ? `${(metres / 1000).toFixed(metres % 1000 ? 1 : 0)} km`
        : `${metres} m`;
    },
    showError(message) {
      const box = $("error");
      box.hidden = false;
      box.textContent = message;
    },
    /** Only the two numbers change while a colour scale eases; keep it cheap. */
    setLegendRange(low, high) {
      const digits = Math.max(high, -low) < 10 ? 2 : 1;
      $("legend-low").textContent = low.toFixed(digits);
      $("legend-high").textContent = high.toFixed(digits);
    },
    setLegend(info) {
      const canvas = $("colorbar");
      const context = canvas.getContext("2d");
      const gradient = context.createLinearGradient(0, 0, canvas.width, 0);
      for (let i = 0; i <= 20; i++) {
        const [r, g, b] = handlers.colour(info.ramp, i / 20);
        gradient.addColorStop(i / 20,
          `rgb(${r | 0}, ${g | 0}, ${b | 0})`);
      }
      context.fillStyle = gradient;
      context.fillRect(0, 0, canvas.width, canvas.height);
      // The numbers are left to setLegendRange: they move while a scale eases,
      // and redrawing this gradient at 60 Hz would be pure waste.
      $("legend-label").textContent = `${info.legend} (${info.unit})`;
    },
  };
}
