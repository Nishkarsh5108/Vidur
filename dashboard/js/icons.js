// Hand-drawn, single-stroke icon set (24x24, currentColor) — no icon font or third-party icon
// package, so the dashboard has no extra network dependency and every glyph stays crisp at any size.
// Usage: icon('pothole', 16) -> an inline <svg> string.

const PATHS = {
  grid: '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
  road: '<path d="M8 3 4 21M16 3l4 18M12 3v3M12 9.5v3M12 16v3"/>',
  cone: '<path d="M12 3l5 15H7z"/><path d="M8.5 13h7M9.7 9.3h4.6M6.5 18h11"/>',
  sign: '<rect x="4" y="7" width="16" height="11" rx="1.5"/><path d="M12 3v4M9 22h6"/><path d="M8 11.5h8M8 14.5h5"/>',
  users: '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17.5" cy="9" r="2.3"/><path d="M15.8 14.2c2.7.3 4.7 2.5 4.7 5.3"/>',
  gauge: '<path d="M4 15a8 8 0 1 1 16 0"/><path d="M12 15l4.2-5.2M4 15h2M18 15h2M12 15V13"/>',
  bus: '<rect x="3.5" y="5" width="17" height="12" rx="2"/><path d="M3.5 11h17M7 17v2M17 17v2"/><circle cx="7.5" cy="19" r="0.1"/>',
  route: '<circle cx="6" cy="18" r="2.3"/><circle cx="18" cy="6" r="2.3"/><path d="M8 18h5a3 3 0 0 0 3-3V9a3 3 0 0 1 3-3"/>',
  building: '<rect x="5" y="3" width="10" height="18" rx="1"/><path d="M15 21h4V11l-4-3M8 7h4M8 11h4M8 15h4"/>',
  shield: '<path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z"/><path d="M9 12l2 2 4-4"/>',
  play: '<path d="M7 4.5v15l13-7.5z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a1.7 1.7 0 0 0 .35 1.9l.05.05a2 2 0 1 1-2.85 2.8l-.06-.06a1.7 1.7 0 0 0-1.9-.34 1.7 1.7 0 0 0-1 1.55V19.6a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.55 1.7 1.7 0 0 0-1.9.34l-.05.06a2 2 0 1 1-2.85-2.8l.05-.05a1.7 1.7 0 0 0 .35-1.9 1.7 1.7 0 0 0-1.55-1H4.4a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.55-1.1 1.7 1.7 0 0 0-.35-1.9l-.05-.05a2 2 0 1 1 2.85-2.85l.05.05a1.7 1.7 0 0 0 1.9.35H10.5a1.7 1.7 0 0 0 1-1.55V4.4a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.55 1.7 1.7 0 0 0 1.9-.35l.05-.05a2 2 0 1 1 2.85 2.85l-.05.05a1.7 1.7 0 0 0-.35 1.9v.05a1.7 1.7 0 0 0 1.55 1h.1a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.55 1z"/>',
  sun: '<circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.3M12 19.2v2.3M4.9 4.9l1.6 1.6M17.5 17.5l1.6 1.6M2.5 12h2.3M19.2 12h2.3M4.9 19.1l1.6-1.6M17.5 6.5l1.6-1.6"/>',
  moon: '<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z"/>',
  chevronLeft: '<path d="M14.5 5.5l-6.5 6.5 6.5 6.5"/>',
  chevronDown: '<path d="M5.5 9l6.5 6.5L18.5 9"/>',
  check: '<path d="M4.5 12.5l5 5 10-10.5"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  alertTriangle: '<path d="M12 4 21.5 20H2.5z"/><path d="M12 10v4.2M12 17.3v.1"/>',
  pin: '<path d="M12 22s7-6.3 7-12a7 7 0 1 0-14 0c0 5.7 7 12 7 12z"/><circle cx="12" cy="10" r="2.4"/>',
  zebra: '<path d="M5 6v12M9 6v12M13 6v12M17 6v12"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14.6-4.4M4 13a8 8 0 0 0 14.6 4.4"/><path d="M20 4v5h-5M4 20v-5h5"/>',
  activity: '<path d="M3 12h4l2.5-7L14 19l2.5-7H21"/>',
  wifiOff: '<path d="M3 3l18 18M5 8.5a15.7 15.7 0 0 1 4-2.3M9.5 16a6 6 0 0 1 7-1M12 20h.01M19 8.5a15.6 15.6 0 0 1 2 1.3M15.5 6.2A15.7 15.7 0 0 1 22 8.5"/>',
};

export function icon(name, size = 16, extra = "") {
  const body = PATHS[name] || PATHS.pin;
  return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" ${extra}>${body}</svg>`;
}
