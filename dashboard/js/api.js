// Backend client. The dashboard is served BY the FastAPI backend (StaticFiles mounted at "/", see
// backend/backend/app/main.py), so every call below is same-origin — no base URL or CORS config
// needed, and it keeps working unmodified whether the page is opened via 127.0.0.1 or a LAN IP.

const BASE = "/api/v1";

function qs(params) {
  if (!params) return "";
  const parts = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "");
  return parts.length ? "?" + new URLSearchParams(parts).toString() : "";
}

async function request(method, path, { params, body } = {}) {
  const res = await fetch(BASE + path + qs(params), {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = (data && (data.detail || data.message)) || res.statusText;
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status);
  }
  return data;
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

const get = (path, params) => request("GET", path, { params });

export const api = {
  health: () => get("/health"),
  kpis: () => get("/kpis"),
  roadHealth: () => get("/road-health"),
  traffic: (hours) => get("/traffic", { hours }),
  bottlenecks: (hours) => get("/traffic/bottlenecks", { hours }),
  missingSigns: () => get("/infrastructure/missing-signs"),

  issues: (params) => get("/issues", params),
  issue: (id) => get(`/issues/${id}`),
  patchIssue: (id, status, note) => request("PATCH", `/issues/${id}`, { body: { status, note: note || null } }),

  vehicles: () => get("/vehicles"),
  vehicle: (id) => get(`/vehicles/${id}`),

  mapEvents: (params) => get("/map/events", params),
  mapIssues: (params) => get("/map/issues", params),
  schoolZones: () => get("/map/school-zones"),

  simulateStatus: () => get("/simulate/status"),
  simulateStart: (body) => request("POST", "/simulate/start", { body }),
  simulateStop: () => request("POST", "/simulate/stop", {}),

  mediaUrl: (id) => `${BASE}/media/${id}`,
};

/** WS /ws/live with automatic reconnect (capped backoff). onMessage receives {type, data}.
 * onState receives 'connecting' | 'open' | 'closed', for the topbar's connection pill. */
export function connectLive({ onMessage, onState }) {
  let socket = null;
  let attempt = 0;
  let closedByCaller = false;

  function connect() {
    if (closedByCaller) return;
    onState?.("connecting");
    const proto = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${proto}://${location.host}/ws/live`);
    socket.onopen = () => { attempt = 0; onState?.("open"); };
    socket.onmessage = (ev) => {
      try { onMessage?.(JSON.parse(ev.data)); } catch { /* ignore a malformed frame */ }
    };
    socket.onclose = () => {
      onState?.("closed");
      if (closedByCaller) return;
      const delay = Math.min(1000 * 2 ** attempt++, 15000);
      setTimeout(connect, delay);
    };
    socket.onerror = () => socket.close();
  }

  connect();
  return { close: () => { closedByCaller = true; socket?.close(); } };
}
