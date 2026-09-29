// The "Simulate" button: starts tools/sim/run_fleet.py on the backend host, polls its status while
// running, and surfaces its log tail — so the demo has one click, but is never a black box.

import { api, ApiError } from "./api.js";
import { icon } from "./icons.js";

const POLL_MS = 1500;

export class SimulateControl {
  constructor({ button, gearButton, popover, logBox, onToast, onFinished }) {
    this.button = button;
    this.popover = popover;
    this.logBox = logBox;
    this.onToast = onToast;
    this.onFinished = onFinished;
    this.running = false;
    this.pollTimer = null;
    this.startedAt = null;

    this.button.onclick = () => (this.running ? this.stop() : this.start());
    gearButton.onclick = (e) => { e.stopPropagation(); popover.classList.toggle("open"); };
    document.addEventListener("click", (e) => { if (!popover.contains(e.target) && e.target !== gearButton) popover.classList.remove("open"); });

    this._render();
    this.refresh();                                  // pick up an already-running simulation on load
  }

  _params() {
    const p = this.popover;
    return {
      rate: Number(p.querySelector("[name=rate]").value),
      clock: p.querySelector("[name=clock]").value.trim() || "now",
      maxConcurrent: Number(p.querySelector("[name=maxConcurrent]").value),
      hud: p.querySelector("[name=hud]").checked,
    };
  }

  async start() {
    this.button.disabled = true;
    try {
      const status = await api.simulateStart(this._params());
      this._apply(status);
      this.onToast("Simulation started — the fleet is loading its models.", "info");
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      this.onToast(`Could not start: ${msg}`, "critical");
    } finally {
      this.button.disabled = false;
    }
  }

  async stop() {
    this.button.disabled = true;
    try {
      this._apply(await api.simulateStop());
      this.onToast("Stopping simulation…", "info");
    } finally {
      this.button.disabled = false;
    }
  }

  async refresh() {
    try {
      this._apply(await api.simulateStatus());
    } catch { /* backend not reachable yet; the topbar's connection pill already reports this */ }
  }

  _apply(status) {
    const wasRunning = this.running;
    this.running = status.running;
    this.startedAt = status.startedAt ? new Date(status.startedAt) : null;
    this.logBox.textContent = (status.log || []).join("\n");
    this.logBox.scrollTop = this.logBox.scrollHeight;
    this._render();

    if (this.running && !this.pollTimer) this.pollTimer = setInterval(() => this.refresh(), POLL_MS);
    if (!this.running && this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
    if (wasRunning && !this.running) {
      this.onToast(status.returnCode ? "Simulation stopped early — see the log." : "Simulation finished.",
        status.returnCode ? "critical" : "good");
      this.onFinished?.();
    }
  }

  _render() {
    this.button.classList.toggle("danger", this.running);
    this.button.classList.toggle("primary", !this.running);
    this.button.innerHTML = this.running
      ? `${icon("stop", 14)} Stop simulation`
      : `${icon("play", 14)} Simulate`;
  }
}
