# FleetSense ops dashboard

A single-page, map-first dashboard for the fleet: potholes, road roughness, traffic, signs, school-zone
alerts and the bus fleet itself, all live. It's served by the [backend](../backend/backend/README.md)
directly — opening `http://localhost:8000/` *is* the dashboard, nothing else to run.

## Why no build step

This is plain HTML, CSS and ES modules — no React, no bundler, no `npm install`. [MapLibre GL JS](https://maplibre.org/)
is vendored locally (`vendor/`) rather than pulled from a CDN, so the one external dependency at
runtime is the base-map tiles themselves (see below), not the mapping library. That was a deliberate
call for a demo that has to "just work" on stage: no dev server to keep alive, no version drift between
a build and what's committed, and the whole app is inspectable by opening the files directly. See
[docs/decisions.md](../docs/decisions.md) for the reasoning (this departs from the Vite/React stack
`docs/dashboard-backend-design.md` originally sketched).

## The "Simulate" button

Click it and the dashboard asks the backend to run [tools/sim/run_fleet.py](../tools/sim/README.md) —
four buses, each a video paired with a real GPS trace, each running the real edge pipeline (V1–V3, S1,
IMU triggers) — and streams the results onto the map as they arrive. The gear icon next to it opens
playback speed, start time and other options; the log panel underneath shows the actual subprocess
output, not a fake progress bar. Requires `python tools/sim/build_scenario.py` to have been run once
first (builds the GPS trails and fetches the real OpenStreetMap school zones).

## Layout

- **Left rail:** switches between six views (Overview, Road Health, Traffic, Infrastructure, Safety,
  Fleet). Each view shows only the 2–4 map layers relevant to it, with its own checkboxes to toggle
  them further — this is the main thing keeping the map from turning into a wall of pins. A compact KPI
  strip (open issues, active alerts, active buses, avg IRI) stays visible underneath regardless of view.
- **Map:** MapLibre GL, CARTO's free Positron/Dark Matter raster basemaps (light/dark, switches with the
  theme). Potholes/signs/zebra crossings are circles sized and coloured by severity/status; traffic
  density is a native MapLibre heatmap layer; bottlenecks pulse; school zones and possibly-missing signs
  are outlined, not filled, so they read as "flagged" rather than "confirmed".
- **Right panel:** a live feed on Overview, ranked/filtered lists on the other views, and a shared issue
  detail card (photo, vote breakdown for signs, Verify/Reject/Resolve) reused everywhere a pin or list
  row is clicked.
- **Ticker:** a thin strip along the bottom — running totals plus the single most recent event, so the
  fleet's activity is visible even with the panel showing something else.

## Live updates

`WS /ws/live` pushes new events, issue updates and vehicle positions straight onto the map and into
whichever list is open. Two endpoints the backend can't push proactively — `/traffic/bottlenecks` and
`/infrastructure/missing-signs`, both computed on demand — are instead polled every ~9 s while their
view is open, and refreshed on a short debounce whenever a relevant WebSocket message arrives.

## Theme

Light / dark / auto (follows the OS), cycled by the sun/moon button and remembered in `localStorage`.
Colours are CSS custom properties (`styles.css`); the map reads them at paint time via `getComputedStyle`
so a theme change doesn't need a second colour table kept in sync by hand.

## Honesty on the map

- The `scenario-note` banner in the corner says plainly that the demo videos are paired with *real* GPS
  traces and *real* OpenStreetMap school zones, not GPS recorded on those actual drives — see
  [tools/sim/README.md](../tools/sim/README.md).
- "Possibly missing signs" and "Bottlenecks" both carry the backend's own `note` field explaining exactly
  what each number does and doesn't mean (see `backend/backend/README.md` §4a/§4b) — surfaced in the
  panel, not smoothed over.
- A photo that failed to reach the backend's media store (or hasn't yet, in a mid-flight demo) shows "No
  photo received for this event" rather than a broken image icon.

## Files

| Path | Role |
|---|---|
| `index.html` | Page shell: topbar, rail, map, panel, ticker, simulate popover |
| `styles.css` | The whole design system: colour tokens (light/dark), layout, every component |
| `js/app.js` | Boots everything; view switching, KPI/rail rendering, live-update handling |
| `js/map.js` | The `FleetMap` class: MapLibre sources/layers, per-view visibility, popups |
| `js/panels.js` | Right-panel list and detail-card rendering |
| `js/simulate.js` | The Simulate button's state machine (start/stop/poll/log) |
| `js/api.js` | Backend REST/WebSocket client (same-origin, no config needed) |
| `js/theme.js`, `js/format.js`, `js/icons.js` | Theme persistence, formatting/type registry, hand-drawn icon set |
| `vendor/` | MapLibre GL JS + CSS, vendored (BSD-licensed) |
