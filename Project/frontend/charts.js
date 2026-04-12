const API_BASE = "/api";

let lineChart = null;
let barChart = null;
const MONTH_ORDER = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

function safeText(id, value) {
  const el = document.getElementById(id);
  if (el) {
    el.textContent = value;
  }
}

function oneDecimalMw(value) {
  const n = Number(value);
  if (Number.isNaN(n)) {
    return "0.0 MW";
  }
  return `${n.toFixed(1)} MW`;
}

function getElementByAnyId(ids) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) {
      return el;
    }
  }
  return null;
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

async function loadSummaryCards() {
  const summary = await fetchJson(`${API_BASE}/summary`);

  safeText("totalStates", summary.total_states ?? "-");
  safeText("energyHubs", summary.energy_hubs ?? "-");
  safeText("energyConsumers", summary.energy_consumers ?? "-");
  safeText("highestOutput", summary.highest_output_mw != null ? Number(summary.highest_output_mw).toFixed(1) : "-");
}

async function loadStateDropdown() {
  const states = await fetchJson(`${API_BASE}/states`);
  const select = getElementByAnyId(["state-select", "stateSelect"]);
  if (!select) {
    return;
  }
  states.sort((a, b) => a.state.localeCompare(b.state));
  select.innerHTML = "";
  states.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.state;
    option.textContent = item.state;
    select.appendChild(option);
  });

  select.addEventListener("change", async (event) => {
    const stateName = event.target.value;
    await updateStatePanel(stateName);
    await updateCharts(stateName);
  });

  if (states.length > 0) {
    const firstState = states[0].state;
    select.value = firstState;
    await updateStatePanel(firstState);
    await updateCharts(firstState);
  }
}

async function updateStatePanel(stateName) {
  const state = await fetchJson(`${API_BASE}/state/${encodeURIComponent(stateName)}`);

  const badgeWrap = getElementByAnyId(["cluster-badge", "clusterBadgeWrap"]);
  if (badgeWrap) {
    const clusterLabel = state.Cluster || state.cluster || "Unknown";
    const isHub = clusterLabel.toLowerCase().includes("hub");
    const bg = isHub ? "#2ecc71" : "#e74c3c";

    badgeWrap.innerHTML = `<span style="display:inline-block;padding:6px 12px;border-radius:999px;color:#fff;font-weight:700;background:${bg};">${clusterLabel}</span>`;
  }

  const totalVal = state.Total_Renewable ?? state.total_renewable;
  const solarVal = state.Solar ?? state.solar;
  const windVal = state.Wind ?? state.wind;
  const biomassVal = state.Biomass ?? state.biomass;
  const hydroVal = state["Small Hydro"] ?? state.small_hydro;

  safeText("sTotal", oneDecimalMw(totalVal));
  safeText("sSolar", oneDecimalMw(solarVal));
  safeText("sWind", oneDecimalMw(windVal));
  safeText("sBiomass", oneDecimalMw(biomassVal));
  safeText("sHydro", oneDecimalMw(hydroVal));
}

async function updateCharts(stateName) {
  const seasonal = await fetchJson(`${API_BASE}/seasonal/${encodeURIComponent(stateName)}`);

  const bucket = {};
  MONTH_ORDER.forEach((m) => {
    bucket[m] = { month: m, Total: 0, Solar: 0, Wind: 0, Biomass: 0, SmallHydro: 0, n: 0 };
  });

  seasonal.forEach((row) => {
    const month = String(row.month || "").toUpperCase().slice(0, 3);
    if (!bucket[month]) {
      return;
    }
    const t = Number(row.Total ?? row.Total_Renewable ?? 0);
    const s = Number(row.Solar ?? 0);
    const w = Number(row.Wind ?? 0);
    const b = Number(row.Biomass ?? 0);
    const h = Number(row.SmallHydro ?? row["Small Hydro"] ?? 0);

    bucket[month].Total += Number.isFinite(t) ? t : 0;
    bucket[month].Solar += Number.isFinite(s) ? s : 0;
    bucket[month].Wind += Number.isFinite(w) ? w : 0;
    bucket[month].Biomass += Number.isFinite(b) ? b : 0;
    bucket[month].SmallHydro += Number.isFinite(h) ? h : 0;
    bucket[month].n += 1;
  });

  const compact = MONTH_ORDER.map((m) => bucket[m])
    .filter((row) => row.n > 0)
    .map((row) => ({
      month: row.month,
      Total: row.Total / row.n,
      Solar: row.Solar / row.n,
      Wind: row.Wind / row.n,
      Biomass: row.Biomass / row.n,
      SmallHydro: row.SmallHydro / row.n,
    }));

  const months = compact.map((row) => row.month);
  const rawTotals = compact.map((row) => Number(row.Total || 0));
  const rawSolar = compact.map((row) => Number(row.Solar || 0));
  const rawWind = compact.map((row) => Number(row.Wind || 0));
  const rawBiomass = compact.map((row) => Number(row.Biomass || 0));
  const rawSmallHydro = compact.map((row) => Number(row.SmallHydro || 0));

  const maxValue = Math.max(...rawTotals, ...rawSolar, ...rawWind, ...rawBiomass, ...rawSmallHydro, 0);
  const scaleFactor = maxValue >= 5000 ? 1000 : 1;
  const unitLabel = scaleFactor === 1000 ? "GW" : "MW";

  const totals = rawTotals.map((v) => v / scaleFactor);
  const solar = rawSolar.map((v) => v / scaleFactor);
  const wind = rawWind.map((v) => v / scaleFactor);
  const biomass = rawBiomass.map((v) => v / scaleFactor);
  const smallHydro = rawSmallHydro.map((v) => v / scaleFactor);
  const maxScaled = Math.max(...totals, ...solar, ...wind, ...biomass, ...smallHydro, 0);

  const lineCanvas = getElementByAnyId(["line-chart", "monthlyTotalChart"]);
  if (lineCanvas) {
    if (lineChart) {
      lineChart.destroy();
    }

    lineChart = new Chart(lineCanvas, {
      type: "line",
      data: {
        labels: months,
        datasets: [
          {
            label: `${stateName} — Total Renewable (MW)`,
            data: totals,
            borderColor: "#1D9E75",
            backgroundColor: "rgba(29, 158, 117, 0.1)",
            fill: true,
            tension: 0.3,
            pointRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        normalized: true,
        plugins: {
          legend: {
            labels: {
              boxWidth: 10,
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            suggestedMax: maxScaled > 0 ? maxScaled * 1.15 : 10,
            title: {
              display: true,
              text: unitLabel,
            },
          },
        },
      },
    });
  }

  const barCanvas = getElementByAnyId(["bar-chart", "monthlyMixChart"]);
  if (barCanvas) {
    if (barChart) {
      barChart.destroy();
    }

    barChart = new Chart(barCanvas, {
      type: "bar",
      data: {
        labels: months,
        datasets: [
          { label: "Solar", data: solar, backgroundColor: "#f39c12" },
          { label: "Wind", data: wind, backgroundColor: "#3498db" },
          { label: "Biomass", data: biomass, backgroundColor: "#27ae60" },
          { label: "SmallHydro", data: smallHydro, backgroundColor: "#8e44ad" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        normalized: true,
        scales: {
          y: {
            stacked: true,
            beginAtZero: true,
            suggestedMax: maxScaled > 0 ? maxScaled * 1.15 : 10,
            title: {
              display: true,
              text: unitLabel,
            },
          },
          x: { stacked: true },
        },
      },
    });
  }
}

async function runPrediction() {
  const ids = [
    "ghi",
    "dni",
    "wind_speed_100m",
    "air_temp",
    "relative_humidity",
    "clearsky_ghi",
    "cloud_opacity",
    "precipitation_rate",
    "albedo",
  ];

  const resultDiv = getElementByAnyId(["prediction-result", "predictionResult"]);
  const payload = {};
  let hasEmpty = false;

  ids.forEach((id) => {
    const input = document.getElementById(id);
    if (!input) {
      return;
    }

    if (input.value.trim() === "") {
      input.style.borderColor = "#e74c3c";
      hasEmpty = true;
    } else {
      input.style.borderColor = "#c9d5d1";
      payload[id] = Number(input.value);
    }
  });

  if (hasEmpty) {
    if (resultDiv) {
      resultDiv.style.color = "#e74c3c";
      resultDiv.textContent = "Please fill all required fields.";
    }
    return;
  }

  try {
    const response = await fetchJson(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (resultDiv) {
      resultDiv.style.color = "#1D9E75";
      resultDiv.style.opacity = "0";
      resultDiv.style.transition = "opacity 300ms ease";
      resultDiv.textContent = `Predicted Output: ${Number(response.predicted_renewable).toFixed(1)} MW`;
      requestAnimationFrame(() => {
        resultDiv.style.opacity = "1";
      });
    }
  } catch (error) {
    if (resultDiv) {
      resultDiv.style.color = "#e74c3c";
      resultDiv.textContent = error.message;
    }
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await loadSummaryCards();
    await loadStateDropdown();
  } catch (error) {
    console.error(error);
  }

  const form = getElementByAnyId(["predict-form", "predictForm"]);
  if (form) {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await runPrediction();
    });
  }
});
