// The map: one MapLibre instance, a handful of GeoJSON sources, and per-view layer visibility.
//
// Base tiles are OpenFreeMap's free, unlimited, no-API-key vector styles (© OpenStreetMap contributors;
// attribution is wired into the map's attribution control, not just this comment). An earlier version of
// this file used CARTO's raster basemaps, which turned out to now require an API key for anonymous use —
// every tile silently came back as a small "API KEY REQUIRED" placeholder image instead of failing loudly,
// so it went unnoticed until someone actually looked at the map. OpenFreeMap's vector styles are MapLibre's
// own recommended free option and ship their own glyphs/sprite, so no extra setup is needed beyond the URL.
// MapLibre's vendored JS (dashboard/vendor/) is the only *code* dependency with no network fetch — the
// basemap tiles themselves are fetched live, exactly like the "free vector style ... for offline use"
// fallback the original design doc (docs/dashboard-backend-design.md §5) already anticipated.
import { cssVar } from "./format.js";

const BASEMAP_STYLE = { light: "https://tiles.openfreemap.org/styles/positron",
                        dark: "https://tiles.openfreemap.org/styles/dark" };

const EMPTY_FC = { type: "FeatureCollection", features: [] };

/** Per-view layer whitelist. Anything not listed here is hidden for that view, which is most of what
 * keeps the map from turning into a wall of pins: only 2-4 data layers are ever visible at once. */
const VIEW_LAYERS = {
  overview: ["potholes", "zebra", "signs", "buses"],
  road: ["iri", "potholes"],
  traffic: ["traffic-heat", "bottlenecks", "buses"],
  infra: ["signs", "zebra", "missing-signs"],
  safety: ["school-zones", "alerts", "buses"],
  fleet: ["buses", "bus-trail"],
};
// Which of a view's layers the rail's checkboxes present (all default on); "buses" is intentionally
// left out of that list on most views — it is supporting context there, a whole nav item (Fleet) once
// it is the point.
export const VIEW_TOGGLEABLE = {
  overview: ["potholes", "zebra", "signs"],
  road: ["iri", "potholes"],
  traffic: ["traffic-heat", "bottlenecks"],
  infra: ["signs", "zebra", "missing-signs"],
  safety: ["school-zones", "alerts"],
  fleet: [],
};

function flatten(fc, mapping) {
  for (const f of fc.features) {
    for (const [target, path] of Object.entries(mapping)) {
      f.properties[target] = path.split(".").reduce((o, k) => (o == null ? o : o[k]), f.properties);
    }
  }
  return fc;
}

function toPointFC(items, toProps) {
  return {
    type: "FeatureCollection",
    features: items.map((item, i) => ({
      type: "Feature",
      id: i,
      geometry: { type: "Point", coordinates: [item.location.longitude, item.location.latitude] },
      properties: toProps(item),
    })),
  };
}

export class FleetMap {
  constructor(container, { onSelectIssue, onSelectVehicle, theme = "light" } = {}) {
    this.onSelectIssue = onSelectIssue;
    this.onSelectVehicle = onSelectVehicle;
    // Set from the real starting theme, not a null placeholder: setTheme()'s `if (theme === this.theme)
    // return` guard only works if this already matches what `style` below actually loads. Getting this
    // wrong meant *every* boot fired a second, redundant setStyle() moments after the first one — which
    // raced against the very first view's data load: MapLibre drops a style's sources the instant
    // setStyle() is called, before the new style has finished loading, so a `getSource(id)?.setData(fc)`
    // landing in that gap silently did nothing (see _set()'s own defensive re-check for the residual risk).
    this.theme = theme;
    this.hiddenByUser = new Set();      // layer ids the rail checkboxes turned off, kept across view switches
    this.view = "overview";
    this.busMarkers = new Map();        // vehicleId -> maplibregl.Marker
    // Our own copy of each source's last data, kept independently of MapLibre's internals, so a
    // theme change (which forces a full setStyle + rebuild) can repopulate every source without
    // re-fetching from the backend or reaching into the GeoJSONSource implementation.
    this._data = {};
    this.map = new maplibregl.Map({
      container,
      style: BASEMAP_STYLE[theme],
      center: [77.15, 28.55],           // Delhi NCR: where the demo fleet's real GPS traces are
      zoom: 10,
      attributionControl: false,
    });
    // OpenFreeMap's style JSON doesn't declare its own source attribution, so it's supplied here —
    // the underlying data is OpenStreetMap's either way, and its licence requires crediting it.
    this.map.addControl(new maplibregl.AttributionControl({
      compact: true, customAttribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
    }), "bottom-right");
    this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    this.popup = new maplibregl.Popup({ closeButton: true, closeOnClick: false, maxWidth: "260px" });

    this._ready = new Promise((resolve) => this.map.on("load", () => { this._build(); resolve(); }));
  }

  ready() { return this._ready; }

  setTheme(theme) {
    if (theme === this.theme) return;
    this.theme = theme;
    this.map.setStyle(BASEMAP_STYLE[theme]);
    this.map.once("style.load", () => this._build());
  }

  // ---- internal: (re)create every source/layer. Runs once on load, and again after each setStyle. ----
  _build() {
    const m = this.map;
    const add = (id, source, type, paint, extra = {}) =>
      m.addLayer({ id, type, source: id, paint, layout: { visibility: "none", ...extra.layout }, ...extra });
    // Re-seeded from our own cache (this._data), not the server: a theme change must not require a
    // network round trip just to keep showing what was already on the map.
    const src = (id) => ({ type: "geojson", data: this._data[id] || EMPTY_FC });

    m.addSource("potholes", src("potholes"));
    m.addSource("iri", src("iri"));
    m.addSource("traffic", src("traffic"));
    m.addSource("bottlenecks", src("bottlenecks"));
    m.addSource("school-zones", src("school-zones"));
    m.addSource("alerts", src("alerts"));
    m.addSource("missing-signs", src("missing-signs"));
    m.addSource("bus-trail", src("bus-trail"));

    // Potholes / road damage: fill by severity, outline by review status.
    add("potholes", "potholes", "circle", {
      "circle-radius": ["+", 5, ["*", 2.6, ["coalesce", ["get", "severity"], 1]]],
      "circle-color": ["match", ["coalesce", ["get", "severity"], 1], 1, cssVar("--fair"), 2, cssVar("--poor"), cssVar("--bad")],
      "circle-opacity": 0.85,
      "circle-stroke-width": 2,
      "circle-stroke-color": ["match", ["get", "status"], "verified", cssVar("--good"), "probable", cssVar("--accent"),
        "rejected", cssVar("--text-faint"), "resolved", cssVar("--text-faint"), cssVar("--info")],
    });
    // Zebra crossings sit on the same "issues" concept but a distinct layer/source so they can be
    // shown or hidden independently (they use the /map/issues?issueType=zebra_crossing source).
    m.addSource("zebra", src("zebra"));
    add("zebra", "zebra", "circle", {
      "circle-radius": 6, "circle-color": cssVar("--info"), "circle-opacity": 0.9,
      "circle-stroke-width": 1.5, "circle-stroke-color": cssVar("--bg-raised"),
    });
    m.addSource("signs", src("signs"));
    add("signs", "signs", "circle", {
      "circle-radius": 6.5,
      "circle-color": ["match", ["get", "_condition"], "damaged", cssVar("--poor"), cssVar("--good")],
      "circle-opacity": 0.9, "circle-stroke-width": 1.5, "circle-stroke-color": cssVar("--bg-raised"),
    });
    add("iri", "iri", "circle", {
      "circle-radius": 4.5,
      "circle-color": ["match", ["get", "_iriClass"], "good", cssVar("--good"), "fair", cssVar("--fair"),
        "poor", cssVar("--poor"), "very_poor", cssVar("--bad"), cssVar("--text-faint")],
      "circle-opacity": 0.75,
    });
    add("traffic-heat", "traffic", "heatmap", {
      "heatmap-weight": ["interpolate", ["linear"], ["coalesce", ["get", "_density"], 1], 0, 0.1, 15, 1],
      "heatmap-intensity": 1.1,
      "heatmap-radius": 22,
      "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], 0, "rgba(0,0,0,0)", 0.2, cssVar("--info"),
        0.5, cssVar("--fair"), 0.75, cssVar("--poor"), 1, cssVar("--bad")],
      "heatmap-opacity": 0.75,
    });
    add("bottlenecks-halo", "bottlenecks", "circle", { "circle-radius": 14, "circle-color": cssVar("--poor"), "circle-opacity": 0.25 });
    add("bottlenecks", "bottlenecks", "circle", {
      "circle-radius": 7, "circle-color": cssVar("--poor"), "circle-stroke-width": 2, "circle-stroke-color": cssVar("--bg-raised"),
    });
    add("school-zones-fill", "school-zones", "fill", { "fill-color": cssVar("--fair"), "fill-opacity": 0.14 });
    add("school-zones-outline", "school-zones", "line", { "line-color": cssVar("--fair"), "line-width": 1.5, "line-dasharray": [2, 2] });
    add("alerts", "alerts", "circle", {
      "circle-radius": 8,
      "circle-color": ["match", ["get", "_level"], "CRITICAL", cssVar("--bad"), cssVar("--fair")],
      "circle-opacity": 0.85, "circle-stroke-width": 2, "circle-stroke-color": cssVar("--bg-raised"),
    });
    add("missing-signs", "missing-signs", "circle", {
      "circle-radius": 9, "circle-color": "rgba(0,0,0,0)", "circle-stroke-width": 2.5, "circle-stroke-color": cssVar("--bad"),
    });
    add("bus-trail", "bus-trail", "line", { "line-color": cssVar("--accent"), "line-width": 2.5, "line-dasharray": [0.2, 1.6] },
      { layout: { "line-cap": "round" } });

    for (const id of ["potholes", "zebra", "signs", "bottlenecks-halo", "bottlenecks", "alerts", "missing-signs"]) {
      m.on("mouseenter", id, () => (m.getCanvas().style.cursor = "pointer"));
      m.on("mouseleave", id, () => (m.getCanvas().style.cursor = ""));
    }
    m.on("click", "potholes", (e) => this._popupIssue(e));
    m.on("click", "zebra", (e) => this._popupIssue(e));
    m.on("click", "signs", (e) => this._popupIssue(e));
    m.on("click", "alerts", (e) => this._popupAlert(e));
    m.on("click", "missing-signs", (e) => this._popupMissingSign(e));

    this._startBottleneckPulse();
    this.applyView(this.view, this.hiddenByUser);
  }

  _popupIssue(e) {
    const p = e.features[0].properties;
    const id = p.id;
    const conf = p.confidence != null ? `${Math.round(p.confidence * 100)}%` : "—";
    this.popup.setLngLat(e.lngLat).setHTML(`
      <div style="padding:10px 12px;min-width:160px">
        <div style="font-weight:700;text-transform:capitalize;font-size:12.5px">${p.issueType.replace(/_/g, " ")}</div>
        <div style="font-size:11px;color:var(--text-dim);margin-top:2px">${p.status} · conf ${conf} · ${p.eventCount} event${p.eventCount === 1 ? "" : "s"}</div>
        <button class="btn ghost" style="margin-top:8px;padding:4px 8px;font-size:11px" data-open>View details →</button>
      </div>`).addTo(this.map);
    this.popup.getElement().querySelector("[data-open]").onclick = () => { this.popup.remove(); this.onSelectIssue?.(id); };
  }

  _popupAlert(e) {
    const p = e.features[0].properties;
    this.popup.setLngLat(e.lngLat).setHTML(`
      <div style="padding:10px 12px;min-width:170px">
        <div style="font-weight:700;font-size:12.5px">${p._level}</div>
        <div style="font-size:11px;color:var(--text-dim);margin-top:2px">${p.metadata?.schoolName ?? "school zone"} · ${p.metadata?.pedestrianCount ?? 0} pedestrian(s)</div>
      </div>`).addTo(this.map);
  }

  _popupMissingSign(e) {
    const p = e.features[0].properties;
    this.popup.setLngLat(e.lngLat).setHTML(`
      <div style="padding:10px 12px;min-width:190px">
        <div style="font-weight:700;font-size:12.5px">Possibly missing sign</div>
        <div style="font-size:11px;color:var(--text-dim);margin-top:2px">last confirmed ${new Date(p.lastConfirmed).toLocaleString()}</div>
        <button class="btn ghost" style="margin-top:8px;padding:4px 8px;font-size:11px" data-open>View issue →</button>
      </div>`).addTo(this.map);
    this.popup.getElement().querySelector("[data-open]").onclick = () => { this.popup.remove(); this.onSelectIssue?.(p.issueId); };
  }

  _startBottleneckPulse() {
    let phase = 0;
    setInterval(() => {
      if (!this.map.getLayer("bottlenecks-halo")) return;
      phase = (phase + 1) % 30;
      this.map.setPaintProperty("bottlenecks-halo", "circle-radius", 12 + 6 * Math.sin(phase / 30 * Math.PI * 2) ** 2);
    }, 90);
  }

  // ---- view / layer visibility ----
  applyView(view, hiddenByUser = this.hiddenByUser) {
    this.view = view;
    this.hiddenByUser = hiddenByUser;
    const allowed = new Set(VIEW_LAYERS[view] || []);
    const layerOf = { potholes: ["potholes"], zebra: ["zebra"], signs: ["signs"], buses: [], iri: ["iri"],
      "traffic-heat": ["traffic-heat"], bottlenecks: ["bottlenecks", "bottlenecks-halo"],
      "missing-signs": ["missing-signs"], "school-zones": ["school-zones-fill", "school-zones-outline"],
      alerts: ["alerts"], "bus-trail": ["bus-trail"] };
    for (const [group, ids] of Object.entries(layerOf)) {
      const visible = allowed.has(group) && !hiddenByUser.has(group);
      for (const id of ids) if (this.map.getLayer(id)) this.map.setLayoutProperty(id, "visibility", visible ? "visible" : "none");
    }
    for (const [id, marker] of this.busMarkers) marker.getElement().style.display = allowed.has("buses") ? "" : "none";
  }

  toggleLayer(group, visible) {
    if (visible) this.hiddenByUser.delete(group); else this.hiddenByUser.add(group);
    this.applyView(this.view, this.hiddenByUser);
  }

  // ---- data setters. Each caches into this._data (so a theme change can rebuild without a re-fetch)
  // and, if the map is mid-rebuild from a theme change, tolerates the source not existing yet: _data is
  // always updated first, so the next _build() picks up the real data even if setData() below can't
  // run right now. The try/catch is a second line of defence — a MapLibre call that threw here (rather
  // than just finding no source) used to abort every render call after it in that view's loader, which
  // is a worse failure than the one line of missing data this guards against. ----
  _set(id, fc) {
    this._data[id] = fc;
    try {
      this.map.getSource(id)?.setData(fc);
    } catch (err) {
      console.warn(`FleetMap: could not update source "${id}" yet (style mid-change); it will catch up on the next rebuild.`, err);
    }
  }
  setPotholes(fc) { this._set("potholes", fc); }
  setZebra(fc) { this._set("zebra", fc); }
  setSigns(fc) {
    for (const f of fc.features) {
      const votes = f.properties.metadata?.conditionVotes || {};
      f.properties._condition = (votes.damaged || 0) > (votes.good || 0) ? "damaged" : "good";
    }
    this._set("signs", fc);
  }
  setIri(fc) { this._set("iri", flatten(fc, { _iriClass: "metadata.iriClass", _iri: "metadata.iri" })); }
  setTraffic(fc) { this._set("traffic", flatten(fc, { _density: "metadata.vehicleCount" })); }
  setAlerts(fc) { this._set("alerts", flatten(fc, { _level: "metadata.level" })); }
  setBottlenecks(list) { this._set("bottlenecks", toPointFC(list, (b) => b)); }
  setMissingSigns(list) { this._set("missing-signs", toPointFC(list, (s) => s)); }
  setSchoolZones(fc) { this._set("school-zones", fc); }

  setBuses(vehicles, activeMinutes) {
    const seen = new Set();
    for (const v of vehicles) {
      seen.add(v.id);
      const loc = v.lastLocation?.coordinates;
      if (!loc) continue;
      const active = v.lastSeenAt && (Date.now() - new Date(v.lastSeenAt).getTime()) / 60000 < activeMinutes;
      let marker = this.busMarkers.get(v.id);
      if (!marker) {
        const el = document.createElement("div");
        el.className = "bus-marker";
        el.innerHTML = `<span class="dot"></span><span class="lbl"></span>`;
        el.onclick = () => this.onSelectVehicle?.(v.id);
        marker = new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat([loc[0], loc[1]]).addTo(this.map);
        this.busMarkers.set(v.id, marker);
      } else {
        marker.setLngLat([loc[0], loc[1]]);
      }
      const el = marker.getElement();
      el.classList.toggle("active", !!active);
      el.querySelector(".lbl").textContent = v.id;
      el.style.display = (VIEW_LAYERS[this.view] || []).includes("buses") ? "" : "none";
    }
    for (const [id, marker] of this.busMarkers) if (!seen.has(id)) { marker.remove(); this.busMarkers.delete(id); }
  }

  setTrail(fc) { this._set("bus-trail", fc || EMPTY_FC); }

  flyTo(lng, lat, zoom = 14) { this.map.flyTo({ center: [lng, lat], zoom: Math.max(zoom, this.map.getZoom()), duration: 900 }); }

  fitTo(fc, padding = 60) {
    if (!fc?.features?.length) return;
    const coords = [];
    for (const f of fc.features) {
      if (f.geometry.type === "Point") coords.push(f.geometry.coordinates);
      else if (f.geometry.type === "LineString") coords.push(...f.geometry.coordinates);
      else if (f.geometry.type === "Polygon") coords.push(...f.geometry.coordinates.flat());
    }
    if (!coords.length) return;
    const bounds = coords.reduce((b, c) => b.extend(c), new maplibregl.LngLatBounds(coords[0], coords[0]));
    this.map.fitBounds(bounds, { padding, duration: 700, maxZoom: 15 });
  }

  /** A brief highlight ring at a location — used when a live WS event arrives, so the map visibly
   * reacts even if the affected layer is currently hidden for the active view. */
  flash(lng, lat, colorVar = "--accent") {
    const el = document.createElement("div");
    el.className = "flash-marker";
    el.style.setProperty("--flash-color", cssVar(colorVar));
    const marker = new maplibregl.Marker({ element: el, anchor: "center" }).setLngLat([lng, lat]).addTo(this.map);
    setTimeout(() => marker.remove(), 1200);
  }
}
