// Light/dark toggle. Three states cycle: auto (follows the OS) -> light -> dark -> auto.
// Persisted in localStorage; falls back to 'auto' silently if storage is unavailable (private window).

const KEY = "vidur.theme";

function read() {
  try { return localStorage.getItem(KEY) || "auto"; } catch { return "auto"; }
}

function write(value) {
  try { localStorage.setItem(KEY, value); } catch { /* private window / storage disabled: theme just won't persist */ }
}

export function applyTheme(value) {
  const root = document.documentElement;
  if (value === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", value);
  write(value);
  return value;
}

export function initTheme() {
  return applyTheme(read());
}

export function cycleTheme() {
  const order = ["auto", "light", "dark"];
  const next = order[(order.indexOf(read()) + 1) % order.length];
  return applyTheme(next);
}

export function currentTheme() {
  return read();
}

/** Resolves 'auto' against the OS preference, for anything (like map tile choice) that needs a concrete light/dark. */
export function effectiveTheme() {
  const value = read();
  if (value !== "auto") return value;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
