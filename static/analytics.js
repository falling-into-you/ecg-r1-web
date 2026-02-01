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

const COUNTRY_CENTROIDS = {
  "CN": [35.8617, 104.1954],
  "US": [37.0902, -95.7129],
  "GB": [55.3781, -3.4360],
  "DE": [51.1657, 10.4515],
  "FR": [46.2276, 2.2137],
  "JP": [36.2048, 138.2529],
  "KR": [35.9078, 127.7669],
  "IN": [20.5937, 78.9629],
  "SG": [1.3521, 103.8198],
  "AU": [-25.2744, 133.7751],
  "CA": [56.1304, -106.3468],
  "BR": [-14.2350, -51.9253],
  "RU": [61.5240, 105.3188],
  "ZA": [-30.5595, 22.9375],
  "IT": [41.8719, 12.5674],
  "ES": [40.4637, -3.7492],
  "NL": [52.1326, 5.2913],
  "SE": [60.1282, 18.6435],
  "CH": [46.8182, 8.2275],
  "MX": [23.6345, -102.5528],
  "AR": [-38.4161, -63.6167],
  "TR": [38.9637, 35.2433],
  "ID": [-0.7893, 113.9213],
  "TH": [15.8700, 100.9925],
  "VN": [14.0583, 108.2772],
  "PH": [12.8797, 121.7740],
  "MY": [4.2105, 101.9758],
  "AE": [23.4241, 53.8478],
  "SA": [23.8859, 45.0792],
  "IL": [31.0461, 34.8516]
};

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

function renderMap(byCountry) {
  markersLayer.clearLayers();
  const items = Array.isArray(byCountry) ? byCountry : [];
  items.forEach(([cc, count]) => {
    if (!cc || cc === "unknown") return;
    const key = String(cc).toUpperCase();
    const center = COUNTRY_CENTROIDS[key];
    if (!center) return;
    const radius = Math.max(6, Math.min(22, 4 + Math.log(count + 1) * 6));
    const marker = L.circleMarker(center, {
      radius,
      weight: 1,
      color: "#1d4ed8",
      fillColor: "#60a5fa",
      fillOpacity: 0.6
    }).bindPopup(`${escapeHtml(key)}: ${escapeHtml(count)}`);
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

  renderTable(el("countryTable"), ["Country", "Requests"], (analytics.by_country || []).slice(0, 50).map(([k, v]) => [k, v]));
  renderTable(el("ipTable"), ["IP", "Requests"], (analytics.by_ip || []).slice(0, 50).map(([k, v]) => [k, v]));
  renderTable(el("recentTable"), ["Timestamp", "IP", "Country", "Feedback", "Request ID"], (analytics.recent || []).map(r => [
    r.timestamp || "",
    r.ip || "",
    r.country || "",
    r.feedback || "",
    r.request_id || ""
  ]));

  renderMap(analytics.by_country || []);
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
