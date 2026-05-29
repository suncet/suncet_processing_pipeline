"""Small standard-library HTTP/SSE dashboard for realtime telemetry."""

from __future__ import annotations

import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .telemetry_state import TelemetryStore


class DashboardServer:
    def __init__(
        self,
        *,
        store: TelemetryStore,
        host: str,
        port: int,
        heartbeat_seconds: float = 15.0,
    ) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.heartbeat_seconds = heartbeat_seconds
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        handler = self._handler_class()
        self._httpd = QuietThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="suncet-realtime-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        store = self.store
        heartbeat = self.heartbeat_seconds

        class Handler(BaseHTTPRequestHandler):
            server_version = "SunCETRealtimeDashboard/0.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/":
                    self._send_html()
                elif path == "/api/snapshot":
                    self._send_json(store.snapshot())
                elif path == "/events":
                    self._send_events()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path == "/api/clear":
                    store.clear_history()
                    self._send_json({"ok": True})
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def _send_html(self) -> None:
                body = DASHBOARD_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_events(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                version: int | None = None
                while True:
                    try:
                        version, snapshot = store.wait_for_snapshot(version, heartbeat)
                        data = json.dumps(snapshot, separators=(",", ":"))
                        self.wfile.write(f"event: snapshot\ndata: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break

        return Handler


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that ignores normal browser disconnects."""

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(
            exc,
            (
                BrokenPipeError,
                ConnectionAbortedError,
                ConnectionResetError,
            ),
        ):
            return
        super().handle_error(request, client_address)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SunCET Realtime Telemetry</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #151514;
      --panel: #222321;
      --panel-soft: #1b1c1a;
      --ink: #eceee8;
      --muted: #a9afa5;
      --line: #3a3d37;
      --green: #4ec28e;
      --amber: #e0a64a;
      --red: #ef6b6b;
      --cyan: #5bc4d6;
      --violet: #b08bdc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 20px;
      border-bottom: 1px solid var(--line);
      background: #1b1c1a;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    button, input {
      border: 1px solid var(--line);
      background: #252722;
      color: var(--ink);
      border-radius: 6px;
      min-height: 34px;
      font-size: 14px;
    }
    button {
      padding: 0 12px;
      cursor: pointer;
    }
    button.active {
      border-color: var(--cyan);
      color: var(--cyan);
      font-weight: 650;
    }
    .zoom-info {
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
      display: inline-flex;
      align-items: center;
    }
    input {
      padding: 0 10px;
      width: min(260px, 42vw);
    }
    main {
      padding: 18px 20px 28px;
      display: grid;
      gap: 16px;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric, .field {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-width: 0;
    }
    .metric .label, .field .name {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .metric .value {
      margin-top: 4px;
      font-size: 20px;
      font-weight: 700;
      line-height: 1.1;
      overflow-wrap: anywhere;
    }
    .metric.good .value { color: var(--green); }
    .metric.warn .value { color: var(--amber); }
    .metric.bad .value { color: var(--red); }
    .power-summary {
      grid-column: span 2;
    }
    .power-summary .value {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      line-height: 1.2;
      font-weight: 700;
    }
    .power-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      min-width: 0;
      padding: 3px 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: var(--panel-soft);
    }
    .power-chip::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: currentColor;
      flex: 0 0 auto;
    }
    .power-chip.on {
      color: var(--green);
      border-color: color-mix(in srgb, var(--green) 55%, var(--line));
    }
    .power-chip.green {
      color: var(--green);
      border-color: color-mix(in srgb, var(--green) 55%, var(--line));
    }
    .power-chip.off {
      color: var(--red);
      border-color: color-mix(in srgb, var(--red) 60%, var(--line));
    }
    .power-chip.red {
      color: var(--red);
      border-color: color-mix(in srgb, var(--red) 60%, var(--line));
    }
    .power-chip.yellow {
      color: var(--amber);
      border-color: color-mix(in srgb, var(--amber) 55%, var(--line));
    }
    .power-chip.unknown {
      color: var(--amber);
      border-color: color-mix(in srgb, var(--amber) 55%, var(--line));
    }
    .power-age {
      color: var(--muted);
      font-weight: 500;
      white-space: nowrap;
    }
    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 4px;
    }
    .section-title h2 {
      margin: 0;
      font-size: 15px;
      font-weight: 650;
    }
    .fields {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 10px;
    }
    .telemetry-section {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .telemetry-section:first-child { margin-top: 0; }
    .group-title {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .power-pairs {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 10px;
      align-items: start;
    }
    .power-pair {
      display: grid;
      gap: 8px;
      min-width: 0;
    }
    .power-pair-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      line-height: 1.2;
      min-height: 15px;
      overflow-wrap: anywhere;
    }
    .field {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .field .top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
    }
    .field .reading {
      font-size: 24px;
      font-weight: 750;
      color: var(--cyan);
      white-space: nowrap;
    }
    .field.stale .reading { color: var(--amber); }
    .field.status-green .reading { color: var(--green); }
    .field.status-yellow .reading { color: var(--amber); }
    .field.status-red .reading { color: var(--red); }
    .field.status-green { border-color: color-mix(in srgb, var(--green) 45%, var(--line)); }
    .field.status-yellow { border-color: color-mix(in srgb, var(--amber) 55%, var(--line)); }
    .field.status-red { border-color: color-mix(in srgb, var(--red) 65%, var(--line)); }
    .field .meta {
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      overflow-wrap: anywhere;
    }
    canvas {
      width: 100%;
      height: 64px;
      border-top: 1px solid var(--line);
      display: block;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    th, td {
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      background: var(--panel-soft);
      font-weight: 650;
    }
    tr:last-child td { border-bottom: 0; }
    .empty {
      color: var(--muted);
      padding: 28px 2px;
    }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      .status-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      input { width: 100%; }
      .toolbar { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>SunCET Realtime Telemetry</h1>
    <div class="toolbar">
      <input id="filter" type="search" placeholder="Filter fields">
      <button id="pause">Pause</button>
      <button id="reset-zoom">Reset Zoom</button>
      <button id="clear">Clear</button>
      <span id="zoom-info" class="zoom-info"></span>
    </div>
  </header>
  <main>
    <section class="status-grid" id="status"></section>
    <section>
      <div class="section-title">
        <h2>Selected Telemetry</h2>
        <span id="field-count" class="empty"></span>
      </div>
      <div id="fields"></div>
      <div class="empty" id="empty">Waiting for matching telemetry.</div>
    </section>
    <section>
      <div class="section-title">
        <h2>Last Packet</h2>
      </div>
      <table>
        <tbody id="last-packet"></tbody>
      </table>
    </section>
  </main>
  <script>
    let paused = false;
    let latestSnapshot = null;
    let zoomRange = null;

    const statusEl = document.getElementById("status");
    const fieldsEl = document.getElementById("fields");
    const emptyEl = document.getElementById("empty");
    const countEl = document.getElementById("field-count");
    const filterEl = document.getElementById("filter");
    const pauseEl = document.getElementById("pause");
    const resetZoomEl = document.getElementById("reset-zoom");
    const clearEl = document.getElementById("clear");
    const zoomInfoEl = document.getElementById("zoom-info");
    const lastPacketEl = document.getElementById("last-packet");

    pauseEl.addEventListener("click", () => {
      paused = !paused;
      pauseEl.classList.toggle("active", paused);
      pauseEl.textContent = paused ? "Resume" : "Pause";
    });
    resetZoomEl.addEventListener("click", () => {
      zoomRange = null;
      render(latestSnapshot);
    });
    clearEl.addEventListener("click", async () => {
      await fetch("/api/clear", {method: "POST"});
    });
    filterEl.addEventListener("input", () => render(latestSnapshot));

    function metric(label, value, cls) {
      return `<div class="metric ${cls || ""}"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></div>`;
    }

    function powerSummary(summary) {
      const states = (summary && summary.states) || [];
      if (!states.length) {
        return `<div class="metric power-summary warn"><div class="label">System Status</div><div class="value">waiting</div></div>`;
      }
      const hasBad = states.some(item => item.state === "off" || item.state === "red");
      const hasWarn = states.some(item => item.state === "unknown" || item.state === "yellow");
      const cls = hasBad ? "bad" : hasWarn ? "warn" : "good";
      const age = Number(summary.age_seconds);
      const ageText = Number.isFinite(age) ? `${age.toFixed(1)} s` : "-";
      const chips = states.map(item => {
        const state = item.state || "unknown";
        const text = item.kind === "mode"
          ? systemStatusLabel(item)
          : `${item.label} ${systemStatusLabel(item)}`;
        return `<span class="power-chip ${escapeAttr(state)}">${escapeHtml(text)}</span>`;
      }).join("");
      return `<div class="metric power-summary ${cls}">
        <div class="label">System Status</div>
        <div class="value">${chips}<span class="power-age">${escapeHtml(ageText)}</span></div>
      </div>`;
    }

    function systemStatusLabel(item) {
      if (item.display_value) return item.display_value;
      const state = item.state || "unknown";
      if (state === "on") return "ON";
      if (state === "off") return "OFF";
      return "?";
    }

    function render(snapshot) {
      if (!snapshot || paused) return;
      latestSnapshot = snapshot;
      const stats = snapshot.stats || {};
      const last = snapshot.last_packet || {};
      const onboardUtc = snapshot.onboard_utc || "";
      const powerStates = snapshot.beacon_power_states || {};
      const age = stats.last_update_wall_time ? snapshot.now - stats.last_update_wall_time : null;
      const linkCls = age === null ? "warn" : age > 10 ? "bad" : "good";
      statusEl.innerHTML = [
        metric("Onboard UTC", onboardUtc || "-", onboardUtc ? "good" : "warn"),
        powerSummary(powerStates),
        metric("Stream", age === null ? "waiting" : `${age.toFixed(1)} s`, linkCls),
        metric("Packets", String(stats.packets_seen || 0), ""),
        metric("Decoded", String(stats.packets_decoded || 0), "good"),
        metric("No Decoder", String(stats.packets_without_decoder || 0), ""),
        metric("Decode Failures", String(stats.decode_failures || 0), stats.decode_failures ? "bad" : ""),
        metric("Selected Points", String(stats.selected_points || 0), ""),
        metric("Filtered Values", String(stats.value_filter_rejections || 0), stats.value_filter_rejections ? "warn" : "")
      ].join("");

      const query = filterEl.value.trim().toLowerCase();
      zoomInfoEl.textContent = zoomRange
        ? `Zoom ${formatJ2000Utc(zoomRange.start)} to ${formatJ2000Utc(zoomRange.end)}`
        : "Drag a plot to zoom time";
      const fields = (snapshot.fields || []).filter(item => {
        if (!query) return true;
        return item.field.toLowerCase().includes(query) || item.packet_name.toLowerCase().includes(query);
      });
      const groups = groupFields(fields);
      const renderedItems = groups.flatMap(group => group.items);
      countEl.textContent = renderedItems.length ? `${renderedItems.length} live fields` : "";
      emptyEl.style.display = renderedItems.length ? "none" : "block";
      fieldsEl.innerHTML = groups.map(groupHtml).join("");
      renderedItems.forEach(item => drawSparkline(item));

      lastPacketEl.innerHTML = [
        row("Packet", last.packet_name || "-"),
        row("APID", last.apid == null ? "-" : String(last.apid)),
        row("Onboard UTC", last.onboard_utc || "-"),
        row("Status", last.decode_status || "-"),
        row("Checksum", last.checksum_status || "-"),
        row("Source", last.source || "-"),
        row("Selected", last.selected_points == null ? "-" : String(last.selected_points)),
        row("Filtered", last.filtered_points == null ? "-" : String(last.filtered_points)),
        row("Error", last.decode_error || "")
      ].join("");
    }

    function fieldHtml(item) {
      const value = Number(item.value);
      const formattedValue = Number.isFinite(value) ? formatTelemetryValue(value) : String(item.value);
      const valueText = `${formattedValue}${item.unit || ""}`;
      const age = Number(item.age_seconds || 0);
      const packetTime = Number(item.packet_time);
      const timeText = item.packet_utc
        ? `UTC ${formatShortUtc(item.packet_utc)}`
        : (Number.isFinite(packetTime) ? `t ${formatTimeSpan(packetTime)}` : "time -");
      const limit = item.limit_status || {state: "unknown", reason: ""};
      const state = ["green", "yellow", "red"].includes(limit.state) ? limit.state : "unknown";
      return `<article class="field ${item.stale ? "stale" : ""} status-${escapeAttr(state)}" data-field="${escapeAttr(item.field)}">
        <div class="top">
          <div class="name">${escapeHtml(item.display_name || item.field)}</div>
          <div class="reading">${escapeHtml(valueText)}</div>
        </div>
        <canvas width="420" height="80"></canvas>
        <div class="meta"><span>${escapeHtml(item.packet_name)}</span><span>${escapeHtml(timeText)}</span><span>${age.toFixed(1)} s old</span></div>
      </article>`;
    }

    function groupHtml(group) {
      if (group.className === "power") {
        return `<section class="telemetry-section">
          <h3 class="group-title">${escapeHtml(group.label)}</h3>
          <div class="power-pairs">${group.pairs.map(powerPairHtml).join("")}</div>
        </section>`;
      }
      return `<section class="telemetry-section">
        <h3 class="group-title">${escapeHtml(group.label)}</h3>
        <div class="fields ${escapeAttr(group.className)}">${group.items.map(fieldHtml).join("")}</div>
      </section>`;
    }

    function powerPairHtml(pair) {
      return `<div class="power-pair">
        <div class="power-pair-title">${escapeHtml(pair.label)}</div>
        ${pair.items.map(fieldHtml).join("")}
      </div>`;
    }

    function pairPowerFields(items) {
      const byKey = new Map();
      items.forEach(item => {
        const key = powerBaseKey(item.field);
        if (!byKey.has(key)) byKey.set(key, []);
        byKey.get(key).push(item);
      });
      return Array.from(byKey.entries())
        .map(([key, groupItems]) => {
          const label = readablePowerLabel(key);
          const sortedItems = groupItems.slice().sort(powerCompare);
          const powerItem = computedPowerItem(key, label, sortedItems);
          if (powerItem) sortedItems.push(powerItem);
          return {key, label, items: sortedItems};
        })
        .sort((a, b) => powerPairRank(a.key) - powerPairRank(b.key) || a.label.localeCompare(b.label));
    }

    function groupFields(fields) {
      const temperatures = [];
      const power = [];
      const other = [];
      fields.forEach(item => {
        const name = item.field.toLowerCase();
        if (isTemperatureField(name)) temperatures.push(item);
        else if (isPowerField(name)) power.push(item);
        else other.push(item);
      });
      temperatures.sort(temperatureCompare);
      power.sort(powerCompare);
      other.sort((a, b) => a.field.localeCompare(b.field));
      const powerPairs = pairPowerFields(power);
      return [
        {label: "Temperatures", className: "temperatures", items: temperatures},
        {label: "Power", className: "power", pairs: powerPairs, items: powerPairs.flatMap(pair => pair.items)},
        {label: "Other", className: "other", items: other},
      ].filter(group => group.items.length > 0);
    }

    function isTemperatureField(name) {
      return name.includes("temp") || name.includes("therm");
    }

    function temperatureCompare(a, b) {
      return temperatureSortKey(a.field).localeCompare(temperatureSortKey(b.field));
    }

    function temperatureSortKey(field) {
      const name = field.toLowerCase();
      if (name === "csie_adc_core_temp") return "csie_adc_core_temp_0";
      if (name === "csie_det0_therm") return "csie_adc_core_temp_1";
      if (name === "csie_det1_therm") return "csie_adc_core_temp_2";
      return name;
    }

    function isPowerField(name) {
      return isVoltageFieldName(name) || isCurrentFieldName(name);
    }

    function isVoltageFieldName(name) {
      return name.includes("volt") || /(^|_)v$/.test(name);
    }

    function isCurrentFieldName(name) {
      return name.includes("curr") || name.includes("current") || /(^|_)i$/.test(name);
    }

    function powerCompare(a, b) {
      const aKey = powerBaseKey(a.field);
      const bKey = powerBaseKey(b.field);
      if (aKey !== bKey) return aKey.localeCompare(bKey);
      return powerMetricRank(a.field) - powerMetricRank(b.field) || a.field.localeCompare(b.field);
    }

    function powerBaseKey(field) {
      return field.toLowerCase()
        .replace(/^beac_/, "")
        .replace(/^ana_/, "")
        .replace(/_volt(age)?$/, "")
        .replace(/_curr(ent)?$/, "")
        .replace(/_charge_current$/, "_charge")
        .replace(/_v$/, "")
        .replace(/_i$/, "")
        .replace(/^batt/, "bat")
        .replace(/_charge$/, "");
    }

    function powerMetricRank(field) {
      const name = field.toLowerCase();
      if (isVoltageFieldName(name)) return 0;
      if (isCurrentFieldName(name)) return 1;
      if (name.endsWith("_power_w")) return 2;
      return 2;
    }

    function computedPowerItem(key, label, items) {
      const voltage = items.find(item => isVoltageFieldName(item.field.toLowerCase()));
      const current = items.find(item => isCurrentFieldName(item.field.toLowerCase()));
      if (!voltage || !current) return null;
      const voltageValue = Number(voltage.value);
      const currentValue = Number(current.value);
      if (!Number.isFinite(voltageValue) || !Number.isFinite(currentValue)) return null;
      const history = computedPowerHistory(voltage.history || [], current.history || []);
      const packetTime = latestFiniteNumber(voltage.packet_time, current.packet_time);
      const packetUtc = latestPacketUtc(voltage, current);
      const receivedTime = latestFiniteNumber(voltage.received_time, current.received_time);
      return {
        field: `${key || "derived"}_power_w`,
        display_name: `${label} Power`,
        unit: " W",
        value: voltageValue * currentValue,
        packet_time: packetTime,
        packet_utc: packetUtc,
        received_time: receivedTime,
        packet_name: "V x I",
        apid: null,
        stale: Boolean(voltage.stale || current.stale),
        age_seconds: Math.max(Number(voltage.age_seconds || 0), Number(current.age_seconds || 0)),
        history,
        limit_status: {
          state: "unknown",
          color: "#5bc4d6",
          reason: "derived from voltage and current",
          limits: {},
        },
      };
    }

    function computedPowerHistory(voltageHistory, currentHistory) {
      const events = [];
      voltageHistory.forEach(point => addPowerHistoryEvent(events, "voltage", point));
      currentHistory.forEach(point => addPowerHistoryEvent(events, "current", point));
      events.sort((a, b) => a.time - b.time || a.rank - b.rank);
      let voltage = null;
      let current = null;
      const history = [];
      for (let index = 0; index < events.length;) {
        const time = events[index].time;
        const batch = [];
        while (index < events.length && Math.abs(events[index].time - time) < 1e-6) {
          batch.push(events[index]);
          index += 1;
        }
        batch.forEach(event => {
          if (event.kind === "voltage") voltage = event.point;
          else current = event.point;
        });
        if (voltage && current) {
          const voltageValue = Number(voltage.value);
          const currentValue = Number(current.value);
          if (Number.isFinite(voltageValue) && Number.isFinite(currentValue)) {
            history.push({
              value: voltageValue * currentValue,
              packet_time: time,
              packet_utc: latestPacketUtc(voltage, current),
              received_time: Math.max(Number(voltage.received_time || 0), Number(current.received_time || 0)),
              packet_name: "V x I",
              limit_status: {
                state: "unknown",
                color: "#5bc4d6",
                reason: "derived from voltage and current",
                limits: {},
              },
            });
          }
        }
      }
      return history;
    }

    function addPowerHistoryEvent(events, kind, point) {
      const time = Number(point.packet_time);
      const value = Number(point.value);
      if (!Number.isFinite(time) || !Number.isFinite(value)) return;
      events.push({kind, point, time, rank: kind === "voltage" ? 0 : 1});
    }

    function latestFiniteNumber(...values) {
      const finite = values.map(Number).filter(Number.isFinite);
      return finite.length ? Math.max(...finite) : null;
    }

    function latestPacketUtc(...items) {
      const stamped = items
        .filter(item => item && item.packet_utc && Number.isFinite(Number(item.packet_time)))
        .sort((a, b) => Number(b.packet_time) - Number(a.packet_time));
      return stamped.length ? stamped[0].packet_utc : "";
    }

    function powerPairRank(key) {
      const ordered = [
        "eps_bus",
        "sa_8_cell_str",
        "sa_9_cell_str",
        "bat1",
        "bat2",
        "uhf",
        "xband",
        "csie",
        "dsps",
      ];
      const index = ordered.indexOf(key);
      return index >= 0 ? index : 1000;
    }

    function readablePowerLabel(key) {
      const specialLabels = {
        "sa_8_cell_str": "SA 8-Cell String",
        "sa_9_cell_str": "SA 9-Cell String",
        "xact": "ADCS / XACT",
        "xband": "X-Band",
      };
      if (specialLabels[key]) return specialLabels[key];
      const acronyms = new Set(["eps", "cdh", "csie", "dsps", "uhf", "xact", "xband", "ifb"]);
      return key.split("_").filter(Boolean).map(part => {
        if (acronyms.has(part)) return part.toUpperCase();
        if (/^\d/.test(part)) return part;
        return part.charAt(0).toUpperCase() + part.slice(1);
      }).join(" ");
    }

    function drawSparkline(item, options) {
      const bind = !options || options.bind !== false;
      const article = fieldsEl.querySelector(`[data-field="${cssEscape(item.field)}"]`);
      if (!article) return;
      const canvas = article.querySelector("canvas");
      const ctx = canvas.getContext("2d");
      const history = item.history || [];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#3a3d37";
      ctx.beginPath();
      ctx.moveTo(0, canvas.height - 18);
      ctx.lineTo(canvas.width, canvas.height - 18);
      ctx.stroke();
      if (history.length < 2) {
        if (bind) bindZoomCanvas(canvas, item, null);
        return;
      }
      const finiteTimeHistory = history.filter(p => Number.isFinite(Number(p.packet_time)) && Number.isFinite(Number(p.value)));
      const times = finiteTimeHistory.map(p => Number(p.packet_time));
      const usePacketTime = times.length >= 2 && Math.min(...times) !== Math.max(...times);
      let plotHistory = usePacketTime
        ? finiteTimeHistory.slice().sort((a, b) => Number(a.packet_time) - Number(b.packet_time))
        : history.filter(p => Number.isFinite(Number(p.value)));
      if (usePacketTime && zoomRange) {
        plotHistory = plotHistory.filter(p => {
          const t = Number(p.packet_time);
          return t >= zoomRange.start && t <= zoomRange.end;
        });
      }
      const values = plotHistory.map(p => Number(p.value)).filter(Number.isFinite);
      if (values.length < 2) {
        drawNoZoomData(ctx, canvas);
        if (bind) bindZoomCanvas(canvas, item, null);
        return;
      }
      const dataMin = Math.min(...values);
      const dataMax = Math.max(...values);
      let min = dataMin;
      let max = dataMax;
      if (min === max) { min -= 1; max += 1; }
      const plotTimes = plotHistory.map(p => Number(p.packet_time));
      const minTime = usePacketTime ? Math.min(...plotTimes) : 0;
      const maxTime = usePacketTime ? Math.max(...plotTimes) : plotHistory.length - 1;
      const maxLabel = formatAxisValue(dataMax);
      const minLabel = formatAxisValue(dataMin);
      ctx.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = "#a9afa5";
      const labelWidth = Math.max(ctx.measureText(maxLabel).width, ctx.measureText(minLabel).width);
      const plotLeft = Math.min(84, Math.max(34, labelWidth + 8));
      const plotRight = 4;
      const plotTop = 8;
      const plotBottom = 16;
      const plotWidth = canvas.width - plotLeft - plotRight;
      const plotHeight = canvas.height - plotTop - plotBottom;
      ctx.textBaseline = "top";
      ctx.fillText(maxLabel, 2, 2);
      ctx.textBaseline = "bottom";
      ctx.fillText(minLabel, 2, canvas.height - 2);
      ctx.strokeStyle = "#3a3d37";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotLeft, plotTop);
      ctx.lineTo(canvas.width - plotRight, plotTop);
      ctx.moveTo(plotLeft, canvas.height - plotBottom);
      ctx.lineTo(canvas.width - plotRight, canvas.height - plotBottom);
      ctx.stroke();
      ctx.strokeStyle = item.stale ? "#e0a64a" : statusColor(item);
      ctx.lineWidth = 2;
      ctx.beginPath();
      plotHistory.forEach((point, index) => {
        const packetTime = Number(point.packet_time);
        const x = usePacketTime
          ? plotLeft + ((packetTime - minTime) / (maxTime - minTime)) * plotWidth
          : plotLeft + index * plotWidth / (plotHistory.length - 1);
        const y = canvas.height - plotBottom - ((Number(point.value) - min) / (max - min)) * plotHeight;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      drawTimeAxis(ctx, canvas, {
        plotLeft,
        plotRight,
        minTime,
        maxTime,
        usePacketTime,
        sampleCount: plotHistory.length,
        rightUtc: usePacketTime ? plotHistory[plotHistory.length - 1].packet_utc : "",
      });
      if (bind) {
        bindZoomCanvas(canvas, item, {
          zoomable: usePacketTime,
          plotLeft,
          plotRight,
          plotWidth,
          minTime,
          maxTime,
        });
      }
    }

    function drawTimeAxis(ctx, canvas, meta) {
      ctx.font = "10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = "#7f8780";
      ctx.textBaseline = "bottom";
      const y = canvas.height - 1;
      const leftLabel = meta.usePacketTime
        ? `span ${formatTimeSpan(meta.maxTime - meta.minTime)}`
        : `last ${meta.sampleCount} samples`;
      const rightLabel = meta.usePacketTime
        ? `UTC ${formatShortUtc(meta.rightUtc) || formatJ2000Utc(meta.maxTime, true)}`
        : "";
      ctx.fillText(leftLabel, meta.plotLeft, y);
      if (!rightLabel) return;
      const leftWidth = ctx.measureText(leftLabel).width;
      const rightWidth = ctx.measureText(rightLabel).width;
      const rightX = canvas.width - meta.plotRight - rightWidth;
      if (rightX > meta.plotLeft + leftWidth + 12) {
        ctx.fillText(rightLabel, rightX, y);
      }
    }

    function bindZoomCanvas(canvas, item, meta) {
      canvas.style.cursor = meta && meta.zoomable ? "crosshair" : "default";
      canvas.onpointerdown = null;
      canvas.onpointermove = null;
      canvas.onpointerup = null;
      canvas.onpointercancel = null;
      canvas.ondblclick = null;
      if (!meta || !meta.zoomable) return;
      canvas.ondblclick = event => {
        event.preventDefault();
        zoomRange = null;
        render(latestSnapshot);
      };
      let dragging = false;
      let startX = 0;
      canvas.onpointerdown = event => {
        dragging = true;
        startX = clamp(canvasEventX(event, canvas), meta.plotLeft, canvas.width - meta.plotRight);
        canvas.setPointerCapture(event.pointerId);
      };
      canvas.onpointermove = event => {
        if (!dragging) return;
        const endX = clamp(canvasEventX(event, canvas), meta.plotLeft, canvas.width - meta.plotRight);
        drawSparkline(item, {bind: false});
        drawSelectionOverlay(canvas, startX, endX);
      };
      canvas.onpointerup = event => {
        if (!dragging) return;
        dragging = false;
        const endX = clamp(canvasEventX(event, canvas), meta.plotLeft, canvas.width - meta.plotRight);
        try { canvas.releasePointerCapture(event.pointerId); } catch (_err) {}
        if (Math.abs(endX - startX) < 6) {
          drawSparkline(item);
          return;
        }
        const t0 = canvasXToTime(startX, meta);
        const t1 = canvasXToTime(endX, meta);
        zoomRange = {start: Math.min(t0, t1), end: Math.max(t0, t1)};
        render(latestSnapshot);
      };
      canvas.onpointercancel = () => {
        dragging = false;
        drawSparkline(item);
      };
    }

    function drawSelectionOverlay(canvas, startX, endX) {
      const ctx = canvas.getContext("2d");
      const left = Math.min(startX, endX);
      const width = Math.abs(endX - startX);
      ctx.fillStyle = "rgba(91, 196, 214, 0.18)";
      ctx.fillRect(left, 0, width, canvas.height);
      ctx.strokeStyle = "#5bc4d6";
      ctx.lineWidth = 1;
      ctx.strokeRect(left, 0.5, width, canvas.height - 1);
    }

    function statusColor(item) {
      const status = item.limit_status || {};
      if (status.color) return status.color;
      if (status.state === "green") return "#4ec28e";
      if (status.state === "yellow") return "#e0a64a";
      if (status.state === "red") return "#ef6b6b";
      return "#5bc4d6";
    }

    function canvasEventX(event, canvas) {
      const rect = canvas.getBoundingClientRect();
      return (event.clientX - rect.left) * canvas.width / rect.width;
    }

    function canvasXToTime(x, meta) {
      const ratio = (x - meta.plotLeft) / meta.plotWidth;
      return meta.minTime + ratio * (meta.maxTime - meta.minTime);
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function drawNoZoomData(ctx, canvas) {
      ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
      ctx.fillStyle = "#a9afa5";
      ctx.textBaseline = "middle";
      ctx.fillText("No points in selected time range", 8, canvas.height / 2);
    }

    function row(label, value) {
      return `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`;
    }

    function formatTimeSpan(seconds) {
      const numeric = Math.abs(Number(seconds));
      if (!Number.isFinite(numeric)) return "-";
      if (numeric < 1) return `${numeric.toFixed(2)}s`;
      if (numeric < 60) return `${numeric < 10 ? numeric.toFixed(1) : numeric.toFixed(0)}s`;
      let remaining = numeric;
      const days = Math.floor(remaining / 86400);
      remaining -= days * 86400;
      const hours = Math.floor(remaining / 3600);
      remaining -= hours * 3600;
      const minutes = Math.floor(remaining / 60);
      const secs = remaining - minutes * 60;
      if (days > 0) return `${days}d ${hours}h`;
      if (hours > 0) return `${hours}h ${minutes}m`;
      return `${minutes}m ${secs.toFixed(0)}s`;
    }

    function formatShortUtc(value) {
      if (!value) return "";
      const match = String(value).match(/T(\d{2}:\d{2}:\d{2})Z$/);
      return match ? `${match[1]}Z` : String(value);
    }

    function formatJ2000Utc(seconds, short = false) {
      const numeric = Number(seconds);
      if (!Number.isFinite(numeric)) return "-";
      const leapSeconds = leapSecondsAfterJ2000(numeric);
      const date = new Date(Date.UTC(2000, 0, 1, 0, 0, 0) + (numeric + leapSeconds) * 1000);
      if (Number.isNaN(date.getTime())) return "-";
      const iso = date.toISOString().replace(/\.\d{3}Z$/, "Z");
      return short ? formatShortUtc(iso) : iso;
    }

    function leapSecondsAfterJ2000(seconds) {
      const epochMs = Date.UTC(2000, 0, 1, 0, 0, 0);
      const effectiveMs = [
        Date.UTC(2006, 0, 1, 0, 0, 0),
        Date.UTC(2009, 0, 1, 0, 0, 0),
        Date.UTC(2012, 6, 1, 0, 0, 0),
        Date.UTC(2015, 6, 1, 0, 0, 0),
        Date.UTC(2017, 0, 1, 0, 0, 0),
      ];
      let leapCount = 0;
      for (let index = 0; index < effectiveMs.length; index += 1) {
        const correctedMs = epochMs + (seconds + leapCount) * 1000;
        const nextCount = effectiveMs.filter(ms => correctedMs >= ms).length;
        if (nextCount === leapCount) break;
        leapCount = nextCount;
      }
      return leapCount;
    }

    function formatAxisValue(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return "-";
      const abs = Math.abs(numeric);
      if (abs >= 1000) return numeric.toExponential(2);
      if (abs >= 100) return numeric.toFixed(1);
      return numeric.toFixed(2);
    }

    function formatTelemetryValue(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return "-";
      return numeric.toFixed(Math.abs(numeric) >= 100 ? 1 : 2);
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
      }[ch]));
    }

    function escapeAttr(value) {
      return escapeHtml(value);
    }

    function cssEscape(value) {
      if (window.CSS && CSS.escape) return CSS.escape(value);
      return String(value).replace(/["\\]/g, "\\$&");
    }

    fetch("/api/snapshot").then(r => r.json()).then(render);
    const events = new EventSource("/events");
    events.addEventListener("snapshot", event => {
      render(JSON.parse(event.data));
    });
  </script>
</body>
</html>
"""
