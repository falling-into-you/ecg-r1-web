function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  const s = String(value ?? "");
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderTable(container, headers, rows) {
  const thead = `<thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`).join("")}</tbody>`;
  container.innerHTML = `<div class="analytics-table-wrap"><table class="analytics-table">${thead}${tbody}</table></div>`;
}

let map = null;
let markersLayer = null;

function initMap() {
  map = L.map("map", { scrollWheelZoom: false });
  map.setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
}

function setStatus(ok, text) {
  const badge = el("analyticsStatus");
  if (!badge) return;
  if (ok) {
    badge.classList.remove("offline");
    badge.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${escapeHtml(text)}`;
  } else {
    badge.classList.add("offline");
    badge.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${escapeHtml(text)}`;
  }
}

async function fetchAnalytics() {
  const resp = await fetch("/admin/analytics_data", { cache: "no-store" });
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}));
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

function renderMap(markers) {
  markersLayer.clearLayers();
  const items = Array.isArray(markers) ? markers : [];
  items.forEach((m) => {
    if (!m) return;
    const lat = Number(m.lat);
    const lon = Number(m.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    const count = Number(m.count || 0);
    const label = m.label || "";
    const radius = Math.max(6, Math.min(22, 4 + Math.log(count + 1) * 6));
    const marker = L.circleMarker([lat, lon], {
      radius,
      weight: 1,
      color: "#1d4ed8",
      fillColor: "#60a5fa",
      fillOpacity: 0.6
    }).bindPopup(`${escapeHtml(label)}: ${escapeHtml(count)}`);
    marker.addTo(markersLayer);
  });
}

function render(analytics) {
  el("totalRequests").textContent = String(analytics.total_requests ?? "--");
  el("uniqueIps").textContent = String(analytics.unique_ips ?? "--");
  const fb = analytics.feedback || {};
  const like = fb.like || 0;
  const dislike = fb.dislike || 0;
  el("feedbackCounts").textContent = `like=${like}, dislike=${dislike}`;

  renderTable(el("countryTable"), ["Region", "Requests"], (analytics.by_region || []).slice(0, 50).map(([k, v]) => [k, v]));
  renderTable(el("ipTable"), ["IP", "Requests"], (analytics.by_ip || []).slice(0, 50).map(([k, v]) => [k, v]));
  renderTable(el("recentTable"), ["Timestamp", "IP", "Region", "City", "Feedback", "Request ID"], (analytics.recent || []).map(r => [
    r.timestamp || "",
    r.ip || "",
    r.region ? `${r.country || ""} / ${r.region}` : (r.country || ""),
    r.city || "",
    r.feedback || "",
    r.request_id || ""
  ]));

  renderMap(analytics.markers || []);
}

async function refresh() {
  try {
    setStatus(false, "Loading");
    const data = await fetchAnalytics();
    render(data);
    setStatus(true, "Ready");
  } catch (e) {
    setStatus(false, `Error`);
    const msg = e && e.message ? e.message : String(e);
    el("countryTable").textContent = msg;
    el("ipTable").textContent = msg;
    el("recentTable").textContent = msg;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  el("refreshBtn").addEventListener("click", refresh);
  refresh();
});
