// Formatting helpers and the single source of truth for how each event/issue type is labelled,
// coloured and iconified across the map, the panels and the ticker.

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function relTime(iso) {
  if (!iso) return "—";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 0) return "just now";
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 86400 * 30) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(iso).toLocaleDateString();
}

export function fmtNum(n, digits = 0) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function titleCase(s) {
  return String(s || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function kmh(mps) {
  return mps === null || mps === undefined ? "—" : `${fmtNum(mps * 3.6, 0)} km/h`;
}

/** issueType / eventType -> presentation. Colours are CSS custom-property names, resolved at paint
 * time via cssVar() so map and UI colours track the active theme without a second colour table. */
export const TYPES = {
  pothole:              { label: "Pothole",         icon: "alertTriangle", color: "--bad" },
  road_damage:          { label: "Road damage",     icon: "alertTriangle", color: "--bad" },
  zebra_crossing:       { label: "Zebra crossing",  icon: "zebra",         color: "--info" },
  traffic_sign:         { label: "Sign",            icon: "sign",         color: "--fair" },
  infrastructure_issue: { label: "Infrastructure",  icon: "building",     color: "--fair" },
  traffic_bottleneck:   { label: "Bottleneck",      icon: "route",        color: "--poor" },
  incident:             { label: "Incident",        icon: "alertTriangle", color: "--bad" },
  traffic_density:      { label: "Traffic sample",  icon: "route",        color: "--info" },
  road_quality:         { label: "Road quality",    icon: "gauge",        color: "--info" },
  pedestrian_alert:     { label: "Pedestrian alert", icon: "users",        color: "--bad" },
  device_health:        { label: "Device health",   icon: "activity",     color: "--text-faint" },
};

export function typeMeta(type) {
  return TYPES[type] || { label: titleCase(type || "event"), icon: "pin", color: "--text-faint" };
}

export function statusBadge(status) {
  return `<span class="badge ${escapeHtml(status)}">${escapeHtml(status)}</span>`;
}

export function levelBadge(level) {
  return `<span class="badge ${level === "CRITICAL" ? "critical" : "advisory"}">${escapeHtml(level)}</span>`;
}
