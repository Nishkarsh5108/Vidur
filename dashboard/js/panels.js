// Right-panel content: lists per view, and the shared issue/vehicle detail card.
// Every render function takes the container element plus data and callbacks — no framework, just
// innerHTML + a pass over the container afterwards to wire up any interactive bits.

import { icon } from "./icons.js";
import { escapeHtml, fmtNum, kmh, relTime, statusBadge, levelBadge, typeMeta } from "./format.js";

function emptyState(message, iconName = "activity") {
  return `<div class="panel-empty">${icon(iconName, 26)}<div>${escapeHtml(message)}</div></div>`;
}

function issueRow(issue) {
  const meta = typeMeta(issue.issueType);
  const conf = issue.confidence != null ? `${Math.round(issue.confidence * 100)}%` : "—";
  return `
    <div class="list-row" data-id="${issue.id}">
      <div class="icon" style="background:color-mix(in srgb, var(${meta.color}) 18%, transparent); color:var(${meta.color})">${icon(meta.icon, 14)}</div>
      <div class="main">
        <div class="title">${escapeHtml(meta.label)} ${statusBadge(issue.status)}</div>
        <div class="meta">
          <span>${issue.eventCount} event${issue.eventCount === 1 ? "" : "s"}</span>
          <span>${issue.vehicleCount} bus${issue.vehicleCount === 1 ? "" : "es"}</span>
          <span>${relTime(issue.lastSeen)}</span>
        </div>
      </div>
      <div class="conf">${conf}</div>
    </div>`;
}

function wireRows(container, onClick) {
  container.querySelectorAll("[data-id]").forEach((row) => { row.onclick = () => onClick(row.dataset.id); });
}

// ---------------- Overview: live feed ----------------

export function renderFeed(container, events) {
  if (!events.length) {
    container.innerHTML = emptyState("No events yet — click Simulate to start the fleet, or wait for live data.", "activity");
    return;
  }
  container.innerHTML = events.map((e) => {
    const meta = typeMeta(e.eventType);
    const conf = e.confidence != null ? `${Math.round(e.confidence * 100)}%` : "";
    return `
      <div class="list-row" data-eid="${e.eventId}">
        <div class="icon" style="background:color-mix(in srgb, var(${meta.color}) 18%, transparent); color:var(${meta.color})">${icon(meta.icon, 14)}</div>
        <div class="main">
          <div class="title">${escapeHtml(meta.label)}</div>
          <div class="meta"><span>${escapeHtml(e.vehicleId)}</span><span>${escapeHtml(e.model)}</span><span>${relTime(e.capturedAt)}</span></div>
        </div>
        <div class="conf">${conf}</div>
      </div>`;
  }).join("");
}

// ---------------- Road health ----------------

export function renderRoadHealth(container, { issues, roadHealth }, onClick) {
  const iri = roadHealth?.iri;
  const header = iri && iri.samples ? `
    <div style="padding:10px var(--space-4);border-bottom:1px solid var(--border)">
      <div style="display:flex;justify-content:space-between;font-size:11.5px;color:var(--text-dim)">
        <span>Avg IRI</span><span class="num">${fmtNum(iri.avgIri, 1)} m/km</span>
      </div>
      <div style="margin-top:6px;display:flex;gap:5px;flex-wrap:wrap">
        ${(iri.byClass || []).map((c) => `<span class="badge" style="background:var(--bg-sunken);color:var(--text-dim)">${c.name.replace("_", " ")} · ${c.count}</span>`).join("")}
      </div>
    </div>` : "";
  if (!issues.length) {
    container.innerHTML = header + emptyState("No pothole or road-damage issues yet.", "road");
    return;
  }
  container.innerHTML = header + issues.map(issueRow).join("");
  wireRows(container, onClick);
}

// ---------------- Traffic ----------------

export function renderTraffic(container, { bottlenecks, summary }, onClick) {
  const classes = summary?.avgByClass || {};
  const total = Object.values(classes).reduce((a, b) => a + (b || 0), 0) || 1;
  const barColors = { carCount: "--info", busCount: "--good", truckCount: "--poor", twoWheelerCount: "--fair", autorickshawCount: "--accent", bicycleCount: "--text-faint" };
  const bar = Object.entries(classes).map(([k, v]) =>
    `<span style="width:${((v || 0) / total) * 100}%;background:var(${barColors[k] || "--text-faint"})" title="${k}: ${fmtNum(v, 1)}"></span>`
  ).join("");
  const header = `
    <div style="padding:10px var(--space-4);border-bottom:1px solid var(--border)">
      <div style="display:flex;justify-content:space-between;font-size:11.5px;color:var(--text-dim)"><span>Avg vehicles / frame</span><span class="num">${fmtNum(summary?.avgVehicleCount, 1)}</span></div>
      <div class="mini-bar">${bar || "<span style=\"width:100%;background:var(--border)\"></span>"}</div>
    </div>`;
  if (!bottlenecks.bottlenecks?.length) {
    container.innerHTML = header + emptyState(bottlenecks.note || "No bottlenecks detected yet.", "route");
    return;
  }
  const rows = bottlenecks.bottlenecks.map((b, i) => `
    <div class="list-row" data-idx="${i}">
      <div class="icon" style="background:color-mix(in srgb, var(--poor) 18%, transparent); color:var(--poor)">${icon("route", 14)}</div>
      <div class="main">
        <div class="title">Bottleneck #${i + 1}</div>
        <div class="meta"><span>${kmh(b.meanSpeedMps)} vs free-flow ${kmh(b.freeFlowSpeedMps)}</span><span>density ${fmtNum(b.meanDensity, 1)}</span></div>
      </div>
      <div class="conf">${fmtNum(b.score, 0)}</div>
    </div>`).join("");
  container.innerHTML = header + rows + `<div style="padding:8px var(--space-4);font-size:10px;color:var(--text-faint)">${escapeHtml(bottlenecks.note)}</div>`;
  container.querySelectorAll("[data-idx]").forEach((row) => {
    row.onclick = () => onClick(bottlenecks.bottlenecks[Number(row.dataset.idx)]);
  });
}

// ---------------- Infrastructure ----------------

export function renderInfrastructure(container, { signs, zebra, missing }, onClick) {
  const section = (title, html) => `<div class="rail-label" style="padding:10px var(--space-4) 4px">${escapeHtml(title)}</div>${html}`;
  const signsHtml = signs.length ? signs.map(issueRow).join("") : emptyState("No signs detected yet.", "sign");
  const missingHtml = missing.count
    ? missing.signs.map((s, i) => `
        <div class="list-row" data-midx="${i}">
          <div class="icon" style="background:color-mix(in srgb, var(--bad) 18%, transparent); color:var(--bad)">${icon("alertTriangle", 14)}</div>
          <div class="main">
            <div class="title">Possibly missing</div>
            <div class="meta"><span>last confirmed ${relTime(s.lastConfirmed)}</span><span>revisited by ${escapeHtml(s.revisitedBy)}</span></div>
          </div>
        </div>`).join("")
    : emptyState(missing.note || "None flagged.", "shield");
  const zebraHtml = zebra.length ? zebra.map(issueRow).join("") : emptyState("No zebra crossings detected yet.", "zebra");

  container.innerHTML = section("Signs", signsHtml) + section("Possibly missing signs", missingHtml) + section("Zebra crossings", zebraHtml);
  wireRows(container, onClick);
  container.querySelectorAll("[data-midx]").forEach((row) => {
    row.onclick = () => onClick(missing.signs[Number(row.dataset.midx)].issueId);
  });
}

// ---------------- Safety ----------------

export function renderSafety(container, alerts, zoneCount) {
  const header = `<div style="padding:8px var(--space-4);font-size:11px;color:var(--text-faint);border-bottom:1px solid var(--border)">${zoneCount} school zone${zoneCount === 1 ? "" : "s"} loaded from OpenStreetMap</div>`;
  if (!alerts.length) {
    container.innerHTML = header + emptyState("No pedestrian alerts yet.", "users");
    return;
  }
  container.innerHTML = header + alerts.map((a) => `
    <div class="list-row">
      <div class="icon" style="background:color-mix(in srgb, var(--bad) 18%, transparent); color:var(--bad)">${icon("users", 14)}</div>
      <div class="main">
        <div class="title">${levelBadge(a.metadata.level)} ${escapeHtml(a.metadata.schoolName || "")}</div>
        <div class="meta"><span>${a.metadata.pedestrianCount ?? 0} pedestrian(s)</span><span>${escapeHtml(a.vehicleId)}</span><span>${relTime(a.capturedAt)}</span></div>
      </div>
    </div>`).join("");
}

// ---------------- Fleet ----------------

export function renderFleet(container, vehicles, activeMinutes, onClick) {
  if (!vehicles.length) {
    container.innerHTML = emptyState("No vehicles yet — run Simulate to bring the demo fleet online.", "bus");
    return;
  }
  container.innerHTML = vehicles.map((v) => {
    const active = v.lastSeenAt && (Date.now() - new Date(v.lastSeenAt).getTime()) / 60000 < activeMinutes;
    return `
      <div class="fleet-row" data-id="${v.id}">
        <div class="head">
          <span class="id">${escapeHtml(v.id)}</span>
          <span class="status-dot" style="background:var(${active ? "--good" : "--text-faint"})"></span>
        </div>
        <div class="meta">${v.eventCount ?? 0} events · last seen ${relTime(v.lastSeenAt)}</div>
      </div>`;
  }).join("");
  container.querySelectorAll("[data-id]").forEach((row) => { row.onclick = () => onClick(row.dataset.id); });
}

export function renderVehicleDetail(container, vehicle, onBack) {
  container.innerHTML = `
    <div class="detail">
      <div class="back" data-back>${icon("chevronLeft", 13)} Fleet</div>
      <h3 style="text-transform:none">${escapeHtml(vehicle.id)}</h3>
      <dl class="kv">
        <dt>Device</dt><dd>${escapeHtml(vehicle.deviceId || "—")}</dd>
        <dt>Events</dt><dd>${fmtNum(vehicle.eventCount)}</dd>
        <dt>Last seen</dt><dd>${relTime(vehicle.lastSeenAt)}</dd>
        <dt>Last trip</dt><dd style="font-family:var(--font-ui);font-size:11px">${escapeHtml(vehicle.lastTripId || "—")}</dd>
      </dl>
      <div class="rail-label">Recent events</div>
      <div>${(vehicle.recentEvents || []).slice(0, 12).map((e) => {
        const meta = typeMeta(e.eventType);
        return `<div class="list-row"><div class="icon" style="background:color-mix(in srgb, var(${meta.color}) 18%, transparent); color:var(${meta.color})">${icon(meta.icon, 13)}</div>
          <div class="main"><div class="title">${escapeHtml(meta.label)}</div><div class="meta">${relTime(e.capturedAt)}</div></div></div>`;
      }).join("") || emptyState("No recent events.")}</div>
    </div>`;
  container.querySelector("[data-back]").onclick = onBack;
}

// ---------------- Issue detail (shared by road / infra / map click) ----------------

export function renderIssueDetail(container, issue, { onBack, onPatch }) {
  const meta = typeMeta(issue.issueType);
  const photoEvent = (issue.events || []).find((e) => e.metadata?.mediaId);
  const votes = issue.metadata?.conditionVotes;
  const voteTotal = votes ? (votes.damaged || 0) + (votes.good || 0) : 0;

  container.innerHTML = `
    <div class="detail">
      <div class="back" data-back>${icon("chevronLeft", 13)} Back</div>
      ${photoEvent
        ? `<img class="photo" src="/api/v1/media/${photoEvent.metadata.mediaId}" onerror="this.outerHTML='<div class=&quot;photo-empty&quot;>No photo received for this event</div>'">`
        : `<div class="photo-empty">No photo attached</div>`}
      <h3>${escapeHtml(meta.label)}</h3>
      ${statusBadge(issue.status)}
      ${votes ? `
        <div style="margin-top:10px">
          <div style="font-size:11px;color:var(--text-dim)">Condition votes — ${votes.damaged || 0} damaged / ${votes.good || 0} good</div>
          <div class="vote-bar">
            <span style="width:${((votes.damaged || 0) / (voteTotal || 1)) * 100}%;background:var(--poor)"></span>
            <span style="width:${((votes.good || 0) / (voteTotal || 1)) * 100}%;background:var(--good)"></span>
          </div>
        </div>` : ""}
      <dl class="kv">
        <dt>Confidence</dt><dd>${issue.confidence != null ? Math.round(issue.confidence * 100) + "%" : "—"}</dd>
        <dt>Severity</dt><dd>${issue.severity ?? "—"}</dd>
        <dt>Events</dt><dd>${issue.eventCount}</dd>
        <dt>Vehicles</dt><dd>${issue.vehicleCount}</dd>
        <dt>Trips</dt><dd>${issue.tripCount}</dd>
        <dt>First seen</dt><dd>${relTime(issue.firstSeen)}</dd>
        <dt>Last seen</dt><dd>${relTime(issue.lastSeen)}</dd>
      </dl>
      ${issue.statusNote ? `<div style="font-size:11.5px;color:var(--text-dim);border-left:2px solid var(--border-strong);padding-left:8px">${escapeHtml(issue.statusNote)}</div>` : ""}
      <textarea placeholder="Note (optional)" data-note></textarea>
      <div class="actions">
        <button class="btn primary" data-status="verified">${icon("check", 13)} Verify</button>
        <button class="btn danger" data-status="rejected">${icon("x", 13)} Reject</button>
        <button class="btn" data-status="resolved">Resolved</button>
      </div>
    </div>`;
  container.querySelector("[data-back]").onclick = onBack;
  container.querySelectorAll("[data-status]").forEach((btn) => {
    btn.onclick = () => onPatch(issue.id, btn.dataset.status, container.querySelector("[data-note]").value);
  });
}
