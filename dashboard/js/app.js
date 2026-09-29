// Bootstraps the dashboard: rail/nav, map, panels, live updates and the Simulate control.
// No framework — a small hand-rolled view model is plenty for six views and a handful of panels.

import { api, connectLive } from "./api.js";
import { icon } from "./icons.js";
import { FleetMap, VIEW_TOGGLEABLE } from "./map.js";
import * as panels from "./panels.js";
import { SimulateControl } from "./simulate.js";
import { currentTheme, cycleTheme, effectiveTheme, initTheme } from "./theme.js";
import { escapeHtml, fmtNum, relTime, typeMeta } from "./format.js";

const $ = (id) => document.getElementById(id);

// A vehicle counts as "active" if it reported within this many minutes — mirrors the backend's own
// ACTIVE_VEHICLE_MINUTES default (backend/backend/app/core/config.py). The API doesn't expose that
// setting, so this is a display-only assumption kept in sync with the documented default.
const ACTIVE_VEHICLE_MINUTES = 10;

const VIEWS = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "road", label: "Road Health", icon: "road" },
  { id: "traffic", label: "Traffic", icon: "route" },
  { id: "infra", label: "Infrastructure", icon: "building" },
  { id: "safety", label: "Safety", icon: "shield" },
  { id: "fleet", label: "Fleet", icon: "bus" },
];
const LAYER_META = {
  potholes: { label: "Potholes / damage", color: "--bad", shape: "round" },
  zebra: { label: "Zebra crossings", color: "--info", shape: "round" },
  signs: { label: "Signs", color: "--fair", shape: "round" },
  iri: { label: "Road roughness", color: "--info", shape: "round" },
  "traffic-heat": { label: "Density heatmap", color: "--poor", shape: "square" },
  bottlenecks: { label: "Bottlenecks", color: "--poor", shape: "round" },
  "missing-signs": { label: "Possibly missing", color: "--bad", shape: "round" },
  "school-zones": { label: "School zones", color: "--fair", shape: "square" },
  alerts: { label: "Pedestrian alerts", color: "--bad", shape: "round" },
};

const state = {
  view: "overview",
  detail: null,          // { kind: 'issue'|'vehicle', id } when the panel shows a detail card
  feed: [],               // recent events, newest first, for the Overview live feed
  vehicles: [],
  hidden: new Set(),
};

let fleetMap;
let simulate;
let pollTimer;
let refreshDebounce;

// ---------------------------------------------------------------- theme

function syncThemeButton() {
  const eff = effectiveTheme();
  $("theme-btn").innerHTML = icon(eff === "dark" ? "moon" : "sun", 15);
  fleetMap?.setTheme(eff);
}

// ---------------------------------------------------------------- toasts + ticker

function toast(message, kind = "info") {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  $("toasts").appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

function tickerLine(text) {
  const el = $("ticker-line");
  el.style.opacity = "0";
  setTimeout(() => { el.textContent = text; el.style.opacity = "1"; }, 160);
}

function describeEvent(e) {
  const meta = typeMeta(e.eventType);
  return `${meta.label} · ${e.vehicleId} · ${relTime(e.capturedAt)}`;
}

// ---------------------------------------------------------------- rail: nav + layers + KPIs

function renderRail(kpis) {
  const issuesByType = Object.fromEntries((kpis?.issuesByType || []).map((r) => [r.name, r.count]));
  const countFor = (view) => ({
    overview: kpis?.openIssues,
    road: (issuesByType.pothole || 0) + (issuesByType.road_damage || 0),
    infra: (issuesByType.traffic_sign || 0) + (issuesByType.zebra_crossing || 0),
    safety: kpis?.activeAlerts,
    fleet: kpis?.activeVehicles,
  }[view]);

  const navHtml = VIEWS.map((v) => {
    const n = countFor(v.id);
    return `
      <button class="nav-item ${v.id === state.view ? "active" : ""}" data-view="${v.id}">
        ${icon(v.icon, 16)}<span>${v.label}</span>
        ${n ? `<span class="count">${n}</span>` : ""}
      </button>`;
  }).join("");

  const layerIds = VIEW_TOGGLEABLE[state.view] || [];
  const layersHtml = layerIds.length ? `
    <div class="rail-section">
      <div class="rail-label">Layers</div>
      ${layerIds.map((id) => {
        const m = LAYER_META[id];
        return `
          <label class="layer-row">
            <input type="checkbox" data-layer="${id}" ${state.hidden.has(id) ? "" : "checked"}>
            <span class="swatch ${m.shape === "round" ? "round" : ""}" style="background:var(${m.color})"></span>
            <span>${m.label}</span>
          </label>`;
      }).join("")}
    </div>` : "";

  const roadQ = kpis?.roadQuality;
  const kpiHtml = `
    <div class="rail-section">
      <div class="rail-label">Fleet status</div>
      <div class="kpi-grid">
        <div class="kpi-tile"><div class="v">${fmtNum(kpis?.openIssues)}</div><div class="l">Open issues</div></div>
        <div class="kpi-tile"><div class="v">${fmtNum(kpis?.activeAlerts)}</div><div class="l">Active alerts</div></div>
        <div class="kpi-tile"><div class="v">${fmtNum(kpis?.activeVehicles)}/${fmtNum(kpis?.totalVehicles)}</div><div class="l">Buses active</div></div>
        <div class="kpi-tile"><div class="v">${roadQ?.avgIri != null ? fmtNum(roadQ.avgIri, 1) : "—"}</div><div class="l">Avg IRI m/km</div></div>
        <div class="kpi-tile wide"><span class="l">Events today</span><span class="v" style="font-size:14px">${fmtNum(kpis?.eventsLast24h)}</span></div>
      </div>
    </div>`;

  $("rail").innerHTML = `<div class="rail-section" style="padding-top:var(--space-2)">${navHtml}</div><hr class="rail-div">${layersHtml}${layersHtml ? '<hr class="rail-div">' : ""}${kpiHtml}`;

  $("rail").querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
  $("rail").querySelectorAll("[data-layer]").forEach((cb) => {
    cb.onchange = () => { fleetMap.toggleLayer(cb.dataset.layer, cb.checked); if (cb.checked) state.hidden.delete(cb.dataset.layer); else state.hidden.add(cb.dataset.layer); };
  });
}

function renderLegend() {
  const layerIds = VIEW_TOGGLEABLE[state.view] || [];
  if (!layerIds.length) { $("map-legend").style.display = "none"; return; }
  $("map-legend").style.display = "";
  $("map-legend").innerHTML = `<h6>Legend</h6>` + layerIds.map((id) => {
    const m = LAYER_META[id];
    return `<div class="row"><span class="swatch ${m.shape === "round" ? "round" : ""}" style="background:var(${m.color})"></span>${m.label}</div>`;
  }).join("");
}

// ---------------------------------------------------------------- panel title + detail navigation

function setPanel(title, hint = "") {
  $("panel-title").textContent = title;
  $("panel-hint").textContent = hint;
}

async function selectIssue(id) {
  state.detail = { kind: "issue", id };
  setPanel("Issue", "");
  try {
    const issue = await api.issue(id);
    panels.renderIssueDetail($("panel-body"), issue, {
      onBack: () => { state.detail = null; renderCurrentView(); },
      onPatch: async (issueId, status, note) => {
        try {
          await api.patchIssue(issueId, status, note);
          toast(`Marked ${status}.`, "good");
          selectIssue(issueId);
        } catch (err) { toast(`Could not update: ${err.message}`, "critical"); }
      },
    });
    if (issue.location) fleetMap.flyTo(issue.location.coordinates[0], issue.location.coordinates[1]);
  } catch (err) {
    $("panel-body").innerHTML = `<div class="panel-empty">Could not load that issue (${escapeHtml(err.message)}).</div>`;
  }
}

async function selectVehicle(id) {
  state.detail = { kind: "vehicle", id };
  setPanel(id, "");
  try {
    const vehicle = await api.vehicle(id);
    panels.renderVehicleDetail($("panel-body"), vehicle, () => { state.detail = null; renderCurrentView(); });
    const loc = vehicle.lastLocation?.coordinates;
    if (loc) fleetMap.flyTo(loc[0], loc[1], 15);
    const events = await api.mapEvents({ vehicleId: id, limit: 400 });
    const coords = events.features.map((f) => f.geometry.coordinates).reverse();
    fleetMap.setTrail(coords.length > 1 ? { type: "Feature", geometry: { type: "LineString", coordinates: coords } } : null);
  } catch (err) {
    $("panel-body").innerHTML = `<div class="panel-empty">Could not load that vehicle (${escapeHtml(err.message)}).</div>`;
  }
}

// ---------------------------------------------------------------- per-view loaders

async function loadOverview() {
  setPanel("Live feed", `${state.feed.length} recent`);
  panels.renderFeed($("panel-body"), state.feed.slice(0, 60));
  const [potholes, zebra, signs] = await Promise.all([
    api.mapIssues({ issueType: "pothole" }), api.mapIssues({ issueType: "zebra_crossing" }), api.mapIssues({ issueType: "traffic_sign" }),
  ]);
  potholes.features.push(...(await api.mapIssues({ issueType: "road_damage" })).features);
  fleetMap.setPotholes(potholes);
  fleetMap.setZebra(zebra);
  fleetMap.setSigns(signs);
}

async function loadRoad() {
  setPanel("Road Health");
  const [potholeFc, damageFc, roadHealth] = await Promise.all([
    api.mapIssues({ issueType: "pothole" }), api.mapIssues({ issueType: "road_damage" }), api.roadHealth(),
  ]);
  const fc = { type: "FeatureCollection", features: [...potholeFc.features, ...damageFc.features] };
  fleetMap.setPotholes(fc);
  const iri = await api.mapEvents({ eventType: "road_quality", limit: 3000 });
  fleetMap.setIri(iri);
  const issues = [...(await api.issues({ issueType: "pothole", limit: 200 })).items, ...(await api.issues({ issueType: "road_damage", limit: 200 })).items]
    .sort((a, b) => (b.severity || 0) * (b.confidence || 0.3) - (a.severity || 0) * (a.confidence || 0.3));
  panels.renderRoadHealth($("panel-body"), { issues, roadHealth }, selectIssue);
}

async function loadTraffic() {
  setPanel("Traffic");
  const [density, bottlenecks, summary] = await Promise.all([
    api.mapEvents({ eventType: "traffic_density", limit: 3000 }), api.bottlenecks(24), api.traffic(24),
  ]);
  fleetMap.setTraffic(density);
  fleetMap.setBottlenecks(bottlenecks.bottlenecks);
  panels.renderTraffic($("panel-body"), { bottlenecks, summary: summary.summary }, (b) => fleetMap.flyTo(b.location.longitude, b.location.latitude, 15));
}

async function loadInfra() {
  setPanel("Infrastructure");
  const [signFc, zebraFc, signIssues, zebraIssues, missing] = await Promise.all([
    api.mapIssues({ issueType: "traffic_sign" }), api.mapIssues({ issueType: "zebra_crossing" }),
    api.issues({ issueType: "traffic_sign", limit: 200 }), api.issues({ issueType: "zebra_crossing", limit: 200 }),
    api.missingSigns(),
  ]);
  fleetMap.setSigns(signFc);
  fleetMap.setZebra(zebraFc);
  fleetMap.setMissingSigns(missing.signs);
  panels.renderInfrastructure($("panel-body"), { signs: signIssues.items, zebra: zebraIssues.items, missing }, selectIssue);
}

async function loadSafety() {
  setPanel("Safety");
  const [alerts, zones] = await Promise.all([api.mapEvents({ eventType: "pedestrian_alert", limit: 500 }), api.schoolZones()]);
  fleetMap.setAlerts(alerts);
  fleetMap.setSchoolZones(zones);
  const alertDocs = (await api.mapEvents({ eventType: "pedestrian_alert", limit: 100 })).features
    .map((f) => ({ ...f.properties, capturedAt: f.properties.capturedAt }))
    .sort((a, b) => new Date(b.capturedAt) - new Date(a.capturedAt));
  panels.renderSafety($("panel-body"), alertDocs, zones.features.length);
  fleetMap.fitTo(zones);
}

async function loadFleet() {
  setPanel("Fleet");
  const vehicles = await api.vehicles();
  state.vehicles = vehicles.items;
  fleetMap.setBuses(state.vehicles, ACTIVE_VEHICLE_MINUTES);
  panels.renderFleet($("panel-body"), state.vehicles, ACTIVE_VEHICLE_MINUTES, selectVehicle);
}

const LOADERS = { overview: loadOverview, road: loadRoad, traffic: loadTraffic, infra: loadInfra, safety: loadSafety, fleet: loadFleet };

// Guards against the same kind of pile-up as scheduleRefresh: without this, a slow load (backend busy
// running the actual simulation) plus a burst of WS-triggered refreshes could have several overlapping
// fetches for the same view in flight at once, each hammering the backend further and finishing in an
// unpredictable order. `viewLoadWanted` still re-runs the loader once more after the in-flight one
// settles, for *whatever view is current at that point* — so switching views mid-load isn't lost, and
// once the burst passes the map/panel end up showing the latest, not a stale interleaved state.
let viewLoadInFlight = false;
let viewLoadWanted = false;

function renderCurrentView() {
  renderLegend();
  viewLoadWanted = true;
  if (viewLoadInFlight) return;
  runViewLoad();
}

async function runViewLoad() {
  viewLoadInFlight = true;
  while (viewLoadWanted) {
    viewLoadWanted = false;
    try {
      await LOADERS[state.view]();
    } catch (err) {
      toast(`Could not load ${state.view}: ${err.message}`, "critical");
    }
  }
  viewLoadInFlight = false;
}

function setView(view) {
  state.view = view;
  state.detail = null;
  fleetMap.applyView(view, state.hidden);
  refreshKpisAndRail();
  renderCurrentView();
}

// ---------------------------------------------------------------- KPIs + polling

async function refreshKpisAndRail() {
  try {
    const kpis = await api.kpis();
    renderRail(kpis);
    $("tick-events").textContent = fmtNum(kpis.totalEvents);
    $("tick-issues").textContent = fmtNum(kpis.openIssues);
    $("tick-vehicles").textContent = fmtNum(kpis.activeVehicles);
  } catch { /* connection pill already reflects backend/DB trouble */ }
}

// Throttled, not debounced: during a fast multi-bus simulation, WS "event" messages can arrive many
// times a second. A plain debounce (reset the timer on every call) would never actually fire until a
// quiet gap — which, at "as fast as possible", might not come until the whole run finishes — so the
// dashboard would look frozen the entire time. This instead guarantees a refresh at least every
// `delay` ms: repeat calls within that window just mark one more refresh as "still wanted".
let refreshWanted = false;

function scheduleRefresh(delay = 1200) {
  refreshWanted = true;
  if (refreshDebounce) return;
  refreshDebounce = setTimeout(() => {
    refreshDebounce = null;
    if (!refreshWanted) return;
    refreshWanted = false;
    refreshKpisAndRail();
    if (!state.detail) renderCurrentView();
  }, delay);
}

// ---------------------------------------------------------------- live updates

function handleLive(message) {
  const { type, data } = message;
  if (type === "event") {
    state.feed.unshift(data);
    state.feed.length = Math.min(state.feed.length, 80);
    tickerLine(describeEvent(data));
    if (data.location) {
      const [lng, lat] = data.location.coordinates;
      fleetMap.flash(lng, lat, typeMeta(data.eventType).color);
    }
    if (state.view === "overview" && !state.detail) panels.renderFeed($("panel-body"), state.feed.slice(0, 60));
    scheduleRefresh();
  } else if (type === "issue_update") {
    if (state.detail?.kind === "issue" && state.detail.id === data.id) selectIssue(data.id);
    scheduleRefresh();
  } else if (type === "vehicle_update") {
    const i = state.vehicles.findIndex((v) => v.id === data.id);
    if (i >= 0) state.vehicles[i] = data; else state.vehicles.push(data);
    fleetMap.setBuses(state.vehicles, ACTIVE_VEHICLE_MINUTES);
    if (state.view === "fleet" && !state.detail) panels.renderFleet($("panel-body"), state.vehicles, ACTIVE_VEHICLE_MINUTES, selectVehicle);
  }
}

function setConnPill(kind) {
  const pill = $("conn-pill");
  pill.className = `pill ${kind === "open" ? "live" : kind === "connecting" ? "busy" : "down"}`;
  $("conn-label").textContent = kind === "open" ? "Live" : kind === "connecting" ? "Connecting…" : "Disconnected";
}

// ---------------------------------------------------------------- boot

async function boot() {
  initTheme();
  // Resolved once, up front, and handed straight to FleetMap: it loads that theme's style as its
  // *first* style load, so there is no second, redundant setStyle() moments later racing the first
  // view's data load (see the comment on FleetMap's constructor for what that race used to break).
  const startTheme = effectiveTheme();
  $("theme-btn").innerHTML = icon(startTheme === "dark" ? "moon" : "sun", 15);
  $("sim-gear").innerHTML = icon("gear", 15);

  fleetMap = new FleetMap("map", { onSelectIssue: selectIssue, onSelectVehicle: selectVehicle, theme: startTheme });
  await fleetMap.ready();
  $("theme-btn").onclick = () => { cycleTheme(); syncThemeButton(); };
  window.matchMedia?.("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (currentTheme() === "auto") syncThemeButton();
  });

  simulate = new SimulateControl({
    button: $("sim-btn"), gearButton: $("sim-gear"), popover: $("sim-popover"), logBox: $("sim-log"),
    onToast: toast, onFinished: () => scheduleRefresh(200),
  });

  try {
    await api.health();
  } catch {
    toast("Backend is unreachable — check that uvicorn and MongoDB are running.", "critical");
  }

  setView("overview");
  connectLive({ onMessage: handleLive, onState: setConnPill });
  pollTimer = setInterval(() => { refreshKpisAndRail(); if (!state.detail && (state.view === "traffic" || state.view === "infra")) renderCurrentView(); }, 9000);
}

boot();
