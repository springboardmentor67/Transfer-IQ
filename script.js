const API_BASE = "https://transfer-iq-infosys.onrender.com";

let players = [];
let currentPlayer = null;

fetch(`${API_BASE}/player_transfer_value_with_sentiment.csv`)
  .then(r => r.text())
  .then(data => {
    const rows    = data.trim().split("\n");
    const headers = rows[0].split(",").map(h => h.trim());
    for (let i = 1; i < rows.length; i++) {
      if (!rows[i].trim()) continue;
      const values = rows[i].split(",");
      let player   = {};
      headers.forEach((header, idx) => { player[header] = (values[idx] || "").trim(); });
      players.push(player);
    }
    populatePlayerDropdown();
  })
  .catch(() => {
    fetch("player_transfer_value_with_sentiment.csv")
      .then(r => r.text())
      .then(data => {
        const rows    = data.trim().split("\n");
        const headers = rows[0].split(",").map(h => h.trim());
        for (let i = 1; i < rows.length; i++) {
          if (!rows[i].trim()) continue;
          const values = rows[i].split(",");
          let player   = {};
          headers.forEach((header, idx) => { player[header] = (values[idx] || "").trim(); });
          players.push(player);
        }
        populatePlayerDropdown();
      })
      .catch(err => console.error("Could not load CSV:", err));
  });


// ── Populate dropdown ──────────────────────────────────────
function populatePlayerDropdown() {
  const select      = document.getElementById("playerSelect");
  const uniqueNames = [...new Set(players.map(p => p.player_name))].sort();
  uniqueNames.forEach(name => {
    const option       = document.createElement("option");
    option.value       = name;
    option.textContent = name;
    select.appendChild(option);
  });
  setupSearch(uniqueNames);
}


// ── Search autocomplete ────────────────────────────────────
function setupSearch(names) {
  const input = document.getElementById("playerSearch");
  const list  = document.getElementById("searchSuggestions");

  input.addEventListener("input", function () {
    const query = this.value.trim().toLowerCase();
    list.innerHTML = "";
    if (!query) { list.style.display = "none"; return; }
    const matches = names.filter(n => n.toLowerCase().includes(query)).slice(0, 8);
    if (!matches.length) { list.style.display = "none"; return; }
    matches.forEach(name => {
      const li  = document.createElement("li");
      const idx = name.toLowerCase().indexOf(query);
      li.innerHTML =
        name.slice(0, idx) +
        "<strong>" + name.slice(idx, idx + query.length) + "</strong>" +
        name.slice(idx + query.length);
      li.addEventListener("mousedown", () => selectPlayer(name));
      list.appendChild(li);
    });
    list.style.display = "block";
  });

  document.addEventListener("click", e => {
    if (!e.target.closest(".search-box-wrapper")) list.style.display = "none";
  });
}


function selectPlayer(name) {
  document.getElementById("playerSearch").value = name;
  document.getElementById("searchSuggestions").style.display = "none";
  const select = document.getElementById("playerSelect");
  select.value = name;
  select.dispatchEvent(new Event("change"));
}


// ── Player selected ────────────────────────────────────────
document.getElementById("playerSelect").addEventListener("change", function () {
  const selectedName  = this.value;
  const seasonWrapper = document.getElementById("seasonWrapper");
  const seasonSelect  = document.getElementById("seasonSelect");
  const detailsDiv    = document.getElementById("playerDetails");
  const hintBox       = document.getElementById("hintBox");
  const predResults   = document.getElementById("predictionResults");

  if (selectedName) document.getElementById("playerSearch").value = selectedName;
  currentPlayer = selectedName || null;

  detailsDiv.style.display  = "none";
  detailsDiv.innerHTML      = "";
  predResults.style.display = "none";
  predResults.innerHTML     = "";
  hintBox.style.display     = selectedName ? "none" : "block";

  if (!selectedName) {
    seasonWrapper.style.display = "none";
    document.getElementById("predictBar").style.display = "none";
    return;
  }

  const seasons = players
    .filter(p => p.player_name === selectedName)
    .map(p => p.season)
    .sort();

  seasonSelect.innerHTML = '<option value="">-- Select a season --</option>';
  seasons.forEach(season => {
    const option       = document.createElement("option");
    option.value       = season;
    option.textContent = season;
    seasonSelect.appendChild(option);
  });

  seasonWrapper.style.display = "block";
  document.getElementById("predictBar").style.display = "flex";
  document.getElementById("predictBarInfo").innerHTML =
    '<strong>' + selectedName + '</strong>Ready to forecast 2024/25 season';
});


// ── Season selected ────────────────────────────────────────
document.getElementById("seasonSelect").addEventListener("change", function () {
  const selectedSeason = this.value;
  const selectedName   = document.getElementById("playerSelect").value;
  const detailsDiv     = document.getElementById("playerDetails");
  const hintBox        = document.getElementById("hintBox");

  if (!selectedSeason) { detailsDiv.style.display = "none"; return; }

  const p = players.find(pl =>
    pl.player_name === selectedName && pl.season === selectedSeason
  );

  if (!p) { detailsDiv.style.display = "none"; return; }

  hintBox.style.display = "none";
  detailsDiv.style.display   = "block";
  detailsDiv.style.animation = "none";
  detailsDiv.offsetHeight;
  detailsDiv.style.animation = "";
  detailsDiv.innerHTML = buildPlayerHTML(p);
});


// ── PREDICT button ─────────────────────────────────────────
function runPredict() {
  if (!currentPlayer) return;

  const btn       = document.getElementById("predictBtn");
  const resultsEl = document.getElementById("predictionResults");

  btn.disabled = true;
  btn.innerHTML = `Running...`;
  resultsEl.style.display = "block";
  resultsEl.innerHTML = `
    <div class="pred-loading">
      <div class="pred-spinner"></div>
      <div class="pred-loading-text">Running prediction for ${currentPlayer}...</div>
      <div class="pred-loading-step" id="loadStep">Initialising models...</div>
    </div>`;

  const steps = [
    "Initialising models...",
    "Univariate forward pass...",
    "Multivariate forward pass...",
    "Encoder-Decoder forecast...",
    "Building XGBoost feature matrix...",
    "Running ensemble prediction...",
    "Finalising results...",
  ];
  let si = 0;
  const stepInterval = setInterval(() => {
    si = Math.min(si + 1, steps.length - 1);
    const el = document.getElementById("loadStep");
    if (el) el.textContent = steps[si];
  }, 380);

  fetch(`${API_BASE}/api/predict/${encodeURIComponent(currentPlayer)}`)
    .then(r => r.json())
    .then(data => {
      clearInterval(stepInterval);
      btn.disabled = false;
      btn.innerHTML = `Predict Next Season 2024/25`;
      if (data.error) {
        resultsEl.innerHTML = `<div class="pred-note">⚠️ ${data.error}</div>`;
        return;
      }
      renderPrediction(data);
    })
    .catch(err => {
      clearInterval(stepInterval);
      btn.disabled = false;
      btn.innerHTML = `Predict Next Season 2024/25`;
      resultsEl.innerHTML = `
      <div class="pred-note">
        ⚠️ Could not reach the server.<br><br>
        This might be because the backend is waking up (Render free tier).  
        Please wait ~30–60 seconds and try again.
      </div>`;
    });
}


// ── Render prediction results ──────────────────────────────
function renderPrediction(data) {
  const resultsEl = document.getElementById("predictionResults");
  const pred      = data.prediction;
  const hist      = data.history;
  const loss      = data.loss_curves;
  const ens       = data.ensemble_predictions || {};
  const ed        = data.encoder_decoder || {};

  // ── Use XGBoost ensemble as the primary market value prediction ──
  // Falls back to LSTM univariate if ensemble model hasn't been run yet.
  const primaryMV = ens.xgb_ensemble || pred.market_value_eur;
  const usingXGB  = !!ens.xgb_ensemble;

  function changeHTML(newVal, oldVal) {
    if (!oldVal || isNaN(newVal) || isNaN(oldVal)) return "";
    const diff = newVal - oldVal;
    const pct  = ((diff / Math.abs(oldVal)) * 100).toFixed(1);
    if (diff > 0)  return `<div class="pred-card-change change-up">▲ +${pct}% vs last</div>`;
    if (diff < 0)  return `<div class="pred-card-change change-down">▼ ${pct}% vs last</div>`;
    return `<div class="pred-card-change change-flat">→ No change</div>`;
  }

  const lastMV  = hist.market_value[hist.market_value.length - 1];
  const predMVM = primaryMV ? (primaryMV / 1e6).toFixed(1) : "—";
  const lastG   = hist.goals[hist.goals.length - 1];
  const lastA   = hist.assists[hist.assists.length - 1];
  const rmseM   = primaryMV ? (Math.abs(primaryMV - (lastMV * 1e6)) / 1e6).toFixed(1) : "—";
  const predSeason = data.predicted_season || "2024/25";
  const allSeasons = [...hist.seasons, `${predSeason} ▶`];

  // ── Stat cards ──
  const cards = [
    { label: "Market Value",   value: `€${predMVM}M`,                             change: changeHTML(primaryMV / 1e6, lastMV) },
    { label: "Goals",          value: Math.round(pred.goals ?? 0),                change: changeHTML(pred.goals, lastG) },
    { label: "Assists",        value: Math.round(pred.assists ?? 0),              change: changeHTML(pred.assists, lastA) },
    { label: "Minutes Played", value: Math.round(pred.minutes_played ?? 0).toLocaleString(), change: "" },
    { label: "Pass Accuracy",  value: `${(pred.pass_accuracy_pct ?? 0).toFixed(1)}%`, change: "" },
    { label: "Injuries",       value: Math.round(pred.total_injuries ?? 0),       change: "" },
    { label: "Availability",   value: `${((pred.availability_rate ?? 0) * 100).toFixed(1)}%`, change: "" },
    { label: "Sentiment Score",value: (pred.vader_compound_score ?? 0).toFixed(3),change: "" },
  ];

  const cardsHTML = cards.map(c => `
    <div class="pred-card">
      <div class="pred-card-label">${c.label}</div>
      <div class="pred-card-value">${c.value}</div>
      ${c.change}
    </div>`).join("");

  // ── Encoder-Decoder row ──
  const edHTML = (ed.step1_eur && ed.step2_eur) ? `
    <div class="section-card ed-forecast-card">
      <h3>Encoder-Decoder Forecast</h3>
      <p class="section-desc">2-season lookahead from the LSTM Encoder-Decoder model</p>
      <div class="ed-steps-row">
        <div class="ed-step-box">
          <div class="ed-step-label">2024/25 (Step 1)</div>
          <div class="ed-step-value">€${(ed.step1_eur / 1e6).toFixed(1)}M</div>
        </div>
        <div class="ed-arrow">→</div>
        <div class="ed-step-box ed-step-box--dim">
          <div class="ed-step-label">2025/26 (Step 2)</div>
          <div class="ed-step-value">€${(ed.step2_eur / 1e6).toFixed(1)}M</div>
        </div>
      </div>
    </div>` : "";

  // ── XGBoost best value card ──
  const xgbVal = ens.xgb_ensemble;
  const xgbHTML = xgbVal ? (() => {
    const xgbM  = (xgbVal / 1e6).toFixed(2);
    const chg   = lastMV ? ((xgbVal / 1e6 - lastMV) / lastMV * 100).toFixed(1) : null;
    const chgColor = chg !== null ? (parseFloat(chg) >= 0 ? "#1a7a4a" : "#e74c3c") : "";
    const chgArrow = chg !== null ? (parseFloat(chg) >= 0 ? "▲" : "▼") : "";
    return `
      <div class="section-card" style="border-top:3px solid #9c27b0;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
          <div>
            <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:var(--text-3);margin-bottom:4px">
              Best Model Prediction
            </div>
            <div style="font-size:0.82rem;font-weight:600;color:#9c27b0;margin-bottom:2px">
              XGBoost + LSTM Ensemble
            </div>
            <div style="font-size:0.74rem;color:var(--text-3)">
              Combines LSTM temporal patterns with XGBoost feature interactions
            </div>
          </div>
          <div style="text-align:right">
            <div style="font-size:2.4rem;font-weight:800;letter-spacing:-1.5px;color:#9c27b0;line-height:1">
              €${xgbM}M
            </div>
            ${chg !== null ? `<div style="font-size:0.82rem;font-weight:600;color:${chgColor};margin-top:3px">${chgArrow} ${Math.abs(chg)}% vs last season</div>` : ""}
          </div>
        </div>
      </div>`;
  })() : "";

  resultsEl.innerHTML = `
    <div class="pred-header">
      <div class="pred-header-left">
        <h2>Model Predictions — ${predSeason}</h2>
        <p>Trained on ${hist.seasons.length} seasons of data for ${data.player}
          &nbsp;·&nbsp;
          <span style="opacity:0.85">Market value via
            <strong>${usingXGB ? "XGBoost + LSTM Ensemble" : "Univariate LSTM"}</strong>
          </span>
        </p>
      </div>
      <div class="pred-header-right">
        <div class="rmse-badge">
          <div class="rmse-label">${usingXGB ? "XGBoost RMSE" : "LSTM RMSE"}</div>
          <div class="rmse-value">€${rmseM}M</div>
        </div>
      </div>
    </div>

    <div class="pred-cards-grid">${cardsHTML}</div>

    ${xgbHTML}

    ${edHTML}

    <div class="pred-charts-row">
      <div class="pred-chart-card">
        <h4>Market Value Trajectory</h4>
        <p>Historical (solid) vs predicted next season (dashed)</p>
        <canvas id="predValueChart"></canvas>
      </div>
      <div class="pred-chart-card">
        <h4>Goals &amp; Assists Trend</h4>
        <p>Historical + predicted performance output</p>
        <canvas id="predPerformChart"></canvas>
      </div>
    </div>

    <div class="pred-charts-row">
      <div class="pred-chart-card">
        <h4>LSTM Loss Curves</h4>
        <p>Training vs validation RMSE over 120 epochs</p>
        <canvas id="predLossChart"></canvas>
      </div>
      <div class="pred-chart-card">
        <h4>Predicted Stats Overview</h4>
        <p>Radar chart comparing key performance metrics</p>
        <canvas id="predRadarChart"></canvas>
      </div>
    </div>
  `;

  setTimeout(() => {
    renderPredLineChart("predValueChart",
      allSeasons,
      [...hist.market_value, null],
      [null, null, null, null, lastMV, parseFloat(predMVM)],
      "Market Value (€M)", "#1a7a4a", "#f0b429"
    );

    const predGoals   = Math.round(pred.goals   ?? 0);
    const predAssists = Math.round(pred.assists  ?? 0);
    renderDoubleLineChart("predPerformChart",
      allSeasons,
      [...hist.goals,   null], [null, null, null, null, lastG, predGoals],
      [...hist.assists, null], [null, null, null, null, lastA, predAssists]
    );

    renderLossChart("predLossChart", loss.epochs, loss.train, loss.val);
    renderRadarChart("predRadarChart", pred, lastG, lastA);
  }, 80);
}


// ── Chart helpers ──────────────────────────────────────────
const predCharts = {};

function destroyPred(id) {
  if (predCharts[id]) { predCharts[id].destroy(); delete predCharts[id]; }
}

const CHART_OPTS = {
  responsive: true,
  maintainAspectRatio: true,
  plugins: {
    legend: { labels: { font: { family: "'Segoe UI', Arial, sans-serif", size: 12 } } },
    tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y ?? ctx.parsed.r}` } }
  },
  scales: {
    x: { grid: { color: "#f0f0f0" }, ticks: { font: { family: "'Segoe UI', Arial" } } },
    y: { beginAtZero: true, grid: { color: "#f0f0f0" }, ticks: { font: { family: "'Segoe UI', Arial" } } }
  }
};

function renderPredLineChart(id, labels, historical, predicted, ylabel, colorH, colorP) {
  destroyPred(id);
  const ctx = document.getElementById(id);
  if (!ctx) return;
  predCharts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Historical", data: historical, borderColor: colorH, backgroundColor: colorH + "18", pointBackgroundColor: colorH, borderWidth: 2.5, pointRadius: 5, fill: true, tension: 0.3, spanGaps: false },
        { label: "Predicted",  data: predicted,  borderColor: colorP, backgroundColor: colorP + "18", pointBackgroundColor: colorP, borderWidth: 2.5, pointRadius: 6, borderDash: [6, 3], fill: false, tension: 0.3, spanGaps: false },
      ]
    },
    options: { ...CHART_OPTS }
  });
}

function renderDoubleLineChart(id, labels, goalsH, goalsP, assiH, assiP) {
  destroyPred(id);
  const ctx = document.getElementById(id);
  if (!ctx) return;
  predCharts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Goals (Actual)",      data: goalsH, borderColor: "#1a7a4a", backgroundColor: "rgba(26,122,74,0.1)", pointRadius: 4, borderWidth: 2.5, tension: 0.3, fill: true, spanGaps: false },
        { label: "Goals (Predicted)",   data: goalsP, borderColor: "#1a7a4a", borderDash: [5,3], pointRadius: 5, borderWidth: 2, tension: 0.3, fill: false, spanGaps: false },
        { label: "Assists (Actual)",    data: assiH, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.08)", pointRadius: 4, borderWidth: 2.5, tension: 0.3, fill: true, spanGaps: false },
        { label: "Assists (Predicted)", data: assiP, borderColor: "#3b82f6", borderDash: [5,3], pointRadius: 5, borderWidth: 2, tension: 0.3, fill: false, spanGaps: false },
      ]
    },
    options: { ...CHART_OPTS }
  });
}

function renderLossChart(id, epochs, train, val) {
  destroyPred(id);
  const ctx = document.getElementById(id);
  if (!ctx) return;
  predCharts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels: epochs,
      datasets: [
        { label: "Train RMSE", data: train, borderColor: "#1a7a4a", backgroundColor: "rgba(26,122,74,0.12)", borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4 },
        { label: "Val RMSE",   data: val,   borderColor: "#f0b429", backgroundColor: "transparent", borderWidth: 2, pointRadius: 0, borderDash: [5,3], tension: 0.4 }
      ]
    },
    options: {
      ...CHART_OPTS,
      scales: {
        x: { ...CHART_OPTS.scales.x, title: { display: true, text: "Epoch", color: "#888" } },
        y: { ...CHART_OPTS.scales.y, title: { display: true, text: "RMSE (scaled)", color: "#888" } }
      }
    }
  });
}

function renderRadarChart(id, pred, lastG, lastA) {
  destroyPred(id);
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const norm = (val, max) => Math.min(100, Math.round((val / max) * 100));
  predCharts[id] = new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Goals", "Assists", "Pass Acc %", "Availability %", "Minutes/90", "Sentiment"],
      datasets: [
        {
          label: "Last Season",
          data: [
            norm(lastG, 50),
            norm(lastA, 30),
            norm(parseFloat(pred.pass_accuracy_pct ?? 80), 100),
            norm((pred.availability_rate ?? 1) * 100, 100),
            norm(pred.minutes_played ?? 2000, 3600),
            norm((pred.vader_compound_score ?? 0) + 1, 2)
          ],
          borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.1)",
          pointBackgroundColor: "#3b82f6", borderWidth: 2
        },
        {
          label: "Predicted 2024/25",
          data: [
            norm(pred.goals   ?? 0, 50),
            norm(pred.assists ?? 0, 30),
            norm(pred.pass_accuracy_pct ?? 80, 100),
            norm((pred.availability_rate ?? 1) * 100, 100),
            norm(pred.minutes_played ?? 2000, 3600),
            norm((pred.vader_compound_score ?? 0) + 1, 2)
          ],
          borderColor: "#1a7a4a", backgroundColor: "rgba(26,122,74,0.15)",
          pointBackgroundColor: "#1a7a4a", borderWidth: 2.5
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { font: { family: "'Segoe UI', Arial", size: 12 } } } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { display: false },
          grid: { color: "#e0e0e0" },
          pointLabels: { font: { family: "'Segoe UI', Arial", size: 11 } }
        }
      }
    }
  });
}


// ═══════════════════════════════════════════════════════════
//  Player stats renderer (unchanged)
// ═══════════════════════════════════════════════════════════
function formatMoney(value) {
  const n = parseFloat(value);
  if (isNaN(n)) return "Unknown";
  if (n >= 1000000) return "€" + (n / 1000000).toFixed(1) + " million";
  if (n >= 1000)    return "€" + (n / 1000).toFixed(0) + ",000";
  return "€" + n;
}

function formatNum(val, decimals) {
  if (decimals === undefined) decimals = 1;
  const n = parseFloat(val);
  if (isNaN(n) || val === "" || val === "nan") return "—";
  return n % 1 === 0 ? n.toLocaleString() : n.toFixed(decimals);
}

function statRow(label, explanation, value) {
  return `
    <div class="stat-row">
      <div class="stat-left">
        <div class="stat-name">${label}</div>
        <div class="stat-explain">${explanation}</div>
      </div>
      <div class="stat-value">${value}</div>
    </div>`;
}

function buildPlayerHTML(p) {
  const pos   = parseFloat(p.positive_count) || 0;
  const neg   = parseFloat(p.negative_count) || 0;
  const neu   = parseFloat(p.neutral_count)  || 0;
  const total = pos + neg + neu || 1;
  const posPct = Math.round(pos / total * 100);
  const negPct = Math.round(neg / total * 100);
  const neuPct = 100 - posPct - negPct;

  const sentLabel    = p.sentiment_label || "Neutral";
  const sentClass    = "sent-" + sentLabel;
  const convRate     = parseFloat(p.shot_conversion_rate);
  const convDisplay  = isNaN(convRate)    ? "—" : (convRate * 100).toFixed(1) + "%";
  const tackleRate   = parseFloat(p.tackle_success_rate);
  const tackleDisplay = isNaN(tackleRate) ? "—" : (tackleRate * 100).toFixed(1) + "%";
  const availRate    = parseFloat(p.availability_rate);
  const availDisplay = isNaN(availRate)   ? "—" : (availRate * 100).toFixed(1) + "%";
  const vaderScore   = parseFloat(p.vader_compound_score) || 0;
  const tbPolarity   = parseFloat(p.tb_polarity) || 0;
  const buzzScore    = parseFloat(p.social_buzz_score) || 0;

  const allSeasons   = players
    .filter(pl => pl.player_name === p.player_name)
    .sort((a, b) => a.season.localeCompare(b.season));

  const chartLabels  = allSeasons.map(s => s.season);
  const chartGoals   = allSeasons.map(s => parseFloat(s.goals)           || 0);
  const chartAssists = allSeasons.map(s => parseFloat(s.assists)          || 0);
  const chartMins    = allSeasons.map(s => parseFloat(s.minutes_played)   || 0);
  const chartValue   = allSeasons.map(s => parseFloat(s.market_value_eur) / 1000000 || 0);

  // ── Market Trend Analysis ──────────────────────────────────
  const mvValues  = allSeasons.map(s => parseFloat(s.market_value_eur) || 0);
  const peakVal   = Math.max(...mvValues);
  const peakSeason = allSeasons[mvValues.indexOf(peakVal)]?.season || "—";
  const firstVal  = mvValues[0] || 1;
  const lastVal   = mvValues[mvValues.length - 1] || 0;
  const totalChg  = ((lastVal - firstVal) / Math.abs(firstVal) * 100).toFixed(1);
  const yoyChanges = mvValues.slice(1).map((v, i) => ({
    season: allSeasons[i + 1]?.season,
    pct: ((v - mvValues[i]) / Math.abs(mvValues[i] || 1) * 100).toFixed(1),
    val: v
  }));
  const recentTrend = yoyChanges.length >= 2
    ? (parseFloat(yoyChanges[yoyChanges.length - 1].pct) + parseFloat(yoyChanges[yoyChanges.length - 2].pct)) / 2
    : parseFloat(yoyChanges[yoyChanges.length - 1]?.pct || 0);
  const trendLabel = recentTrend > 5 ? "Rising" : recentTrend < -5 ? "Declining" : "Stable";
  const trendColor = recentTrend > 5 ? "#1a7a4a" : recentTrend < -5 ? "#e74c3c" : "#f0b429";
  const attractScore = parseFloat(p.transfer_attractiveness_score) || 0;
  const attractPct   = Math.min(100, Math.max(0, attractScore * 10));

  const yoyHTML = yoyChanges.map(y => {
    const pct = parseFloat(y.pct);
    const col = pct > 0 ? "#1a7a4a" : pct < 0 ? "#e74c3c" : "#888";
    const arrow = pct > 0 ? "▲" : pct < 0 ? "▼" : "→";
    return `<div class="yoy-item">
      <span class="yoy-season">${y.season}</span>
      <span class="yoy-val">€${(y.val/1e6).toFixed(1)}M</span>
      <span class="yoy-chg" style="color:${col}">${arrow} ${Math.abs(pct)}%</span>
    </div>`;
  }).join("");

  // ── Sentiment gauge ──────────────────────────────────────
  const vaderPct   = Math.round(((vaderScore + 1) / 2) * 100);
  const vaderColor = vaderScore > 0.05 ? "#1a7a4a" : vaderScore < -0.05 ? "#e74c3c" : "#f0b429";
  const tbPct      = Math.round(((tbPolarity + 1) / 2) * 100);
  const tbColor    = tbPolarity > 0.05 ? "#1a7a4a" : tbPolarity < -0.05 ? "#e74c3c" : "#f0b429";
  const buzzPct    = Math.min(100, Math.round(buzzScore / 10 * 100));

  setTimeout(() => {
    renderChart(chartLabels, chartGoals, chartAssists, chartMins, chartValue);
    renderSentimentChart(posPct, negPct, neuPct);
  }, 50);

  return `
    <!-- Player Header -->
    <div class="player-header-card">
      <div class="ph-left">
        <div class="ph-name">${p.player_name}</div>
        <div class="ph-tags">
          <span class="tag">${p.team || "Unknown Team"}</span>
          <span class="tag">${p.position || "Unknown Position"}</span>
          <span class="tag">Season ${p.season || "—"}</span>
          <span class="tag">Age ${p.current_age || "—"}</span>
          <span class="tag">${p.career_stage || "—"}</span>
        </div>
      </div>
      <div class="ph-right">
        <div class="ph-value-label">Market Value</div>
        <div class="ph-value">${formatMoney(p.market_value_eur)}</div>
        <div class="ph-trend" style="color:${trendColor}">${trendLabel}</div>
      </div>
    </div>

    <!-- Performance Over Seasons -->
    <div class="section-card">
      <div class="section-header">
        <h3>Performance Over Seasons</h3>
        <div class="chart-stat-toggle">
          <button class="chart-btn active" onclick="switchStat('market_value', this)">Market Value</button>
          <button class="chart-btn" onclick="switchStat('goals_assists', this)">Goals &amp; Assists</button>
          <button class="chart-btn" onclick="switchStat('minutes', this)">Minutes</button>
        </div>
      </div>
      <div class="chart-wrapper">
        <canvas id="playerChart"></canvas>
      </div>
    </div>

    <!-- Market Analysis Trend -->
    <div class="section-card">
      <div class="section-header">
        <h3>Market Value Analysis</h3>
        <span class="section-badge" style="background:${trendColor}22;color:${trendColor}">${trendLabel}</span>
      </div>
      <div class="market-analysis-grid">
        <div class="ma-stat">
          <div class="ma-stat-label">Current Value</div>
          <div class="ma-stat-value">€${(lastVal/1e6).toFixed(1)}M</div>
        </div>
        <div class="ma-stat">
          <div class="ma-stat-label">Peak Value</div>
          <div class="ma-stat-value">€${(peakVal/1e6).toFixed(1)}M</div>
          <div class="ma-stat-sub">${peakSeason}</div>
        </div>
        <div class="ma-stat">
          <div class="ma-stat-label">5-Season Change</div>
          <div class="ma-stat-value" style="color:${parseFloat(totalChg)>=0?'#1a7a4a':'#e74c3c'}">${parseFloat(totalChg)>=0?'▲':'▼'} ${Math.abs(totalChg)}%</div>
        </div>
        <div class="ma-stat">
          <div class="ma-stat-label">Transfer Attractiveness</div>
          <div class="ma-attract-bar">
            <div class="ma-attract-fill" style="width:${attractPct}%;background:${attractScore>6?'#1a7a4a':attractScore>3?'#f0b429':'#e74c3c'}"></div>
          </div>
          <div class="ma-stat-sub">${formatNum(p.transfer_attractiveness_score, 2)} / 10</div>
        </div>
      </div>
      <div class="yoy-section">
        <div class="yoy-label">Year-on-Year Value Change</div>
        <div class="yoy-list">${yoyHTML}</div>
      </div>
    </div>

    <!-- On-Pitch Performance -->
    <div class="section-card">
      <h3>On-Pitch Performance</h3>
      ${statRow("Matches Played",       "Total games this season",                formatNum(p.matches, 0))}
      ${statRow("Minutes Played",       "Total time on the pitch",                (parseFloat(p.minutes_played)||0).toLocaleString() + " mins")}
      ${statRow("Goals",                "Times they scored",                      formatNum(p.goals, 0))}
      ${statRow("Assists",              "Times they helped someone score",        formatNum(p.assists, 0))}
      ${statRow("Goals per 90 mins",    "How often they score per full game",     formatNum(p.goals_per90, 2))}
      ${statRow("Assists per 90 mins",  "How often they assist per full game",    formatNum(p.assists_per90, 2))}
      ${statRow("Shot Conversion Rate", "% of shots that become goals",           convDisplay)}
    </div>

    <!-- Passing & Defending -->
    <div class="section-two-col">
      <div class="section-card">
        <h3>Passing</h3>
        ${statRow("Total Passes",      "All passes attempted",            formatNum(p.passes_total, 0))}
        ${statRow("Successful Passes", "Passes that reached a teammate",  formatNum(p.passes_complete, 0))}
        ${statRow("Pass Accuracy",     "% of passes successful",          formatNum(p.pass_accuracy_pct, 1) + "%")}
      </div>
      <div class="section-card">
        <h3>Defending</h3>
        ${statRow("Tackles Made",        "Times challenged for the ball",   formatNum(p.tackles_total, 0))}
        ${statRow("Tackles Won",         "Tackles where they got the ball", formatNum(p.tackles_won, 0))}
        ${statRow("Tackle Success Rate", "% of tackles successful",         tackleDisplay)}
        ${statRow("Interceptions",       "Times they cut off a pass",       formatNum(p.interceptions, 0))}
      </div>
    </div>

    <!-- Injury Record -->
    <div class="section-card">
      <h3>Injury Record</h3>
      ${statRow("Total Injuries",    "Number of injuries this season",   formatNum(p.total_injuries, 0))}
      ${statRow("Days Injured",      "Total days missed due to injury",  formatNum(p.total_days_injured, 0) + " days")}
      ${statRow("Matches Missed",    "Games they could not play",        formatNum(p.total_matches_missed, 0))}
      ${statRow("Availability Rate", "% of games they were fit to play", availDisplay)}
      ${p.most_common_injury && p.most_common_injury !== "" && p.most_common_injury !== "nan"
        ? statRow("Most Common Injury", "Most frequent injury type", p.most_common_injury) : ""}
    </div>

    <!-- Sentiment Analysis -->
    <div class="section-card">
      <div class="section-header">
        <h3>Sentiment Analysis</h3>
        <span class="section-badge ${sentClass}">${sentLabel}</span>
      </div>

      <div class="sentiment-analysis-layout">

        <!-- Donut chart -->
        <div class="sa-donut-wrap">
          <div class="donut-wrapper"><canvas id="sentimentChart"></canvas></div>
          <div class="sa-donut-legend">
            <div class="sa-legend-item"><span class="sa-dot" style="background:#2ecc71"></span>Positive <strong>${posPct}%</strong></div>
            <div class="sa-legend-item"><span class="sa-dot" style="background:#e74c3c"></span>Negative <strong>${negPct}%</strong></div>
            <div class="sa-legend-item"><span class="sa-dot" style="background:#bdc3c7"></span>Neutral <strong>${neuPct}%</strong></div>
          </div>
        </div>

        <!-- Score bars -->
        <div class="sa-scores">
          <div class="sa-score-row">
            <div class="sa-score-label">VADER Compound Score</div>
            <div class="sa-score-track">
              <div class="sa-score-fill" style="width:${vaderPct}%;background:${vaderColor}"></div>
              <div class="sa-score-mid"></div>
            </div>
            <div class="sa-score-val" style="color:${vaderColor}">${vaderScore.toFixed(3)}</div>
          </div>
          <div class="sa-score-row">
            <div class="sa-score-label">TextBlob Polarity</div>
            <div class="sa-score-track">
              <div class="sa-score-fill" style="width:${tbPct}%;background:${tbColor}"></div>
              <div class="sa-score-mid"></div>
            </div>
            <div class="sa-score-val" style="color:${tbColor}">${tbPolarity.toFixed(3)}</div>
          </div>
          <div class="sa-score-row">
            <div class="sa-score-label">Social Buzz Score</div>
            <div class="sa-score-track sa-score-track--full">
              <div class="sa-score-fill" style="width:${buzzPct}%;background:#3b82f6"></div>
            </div>
            <div class="sa-score-val" style="color:#3b82f6">${formatNum(p.social_buzz_score, 2)}</div>
          </div>

          <div class="sa-stats-row">
            ${statRow("Total Tweets",  "Mentions on social media",               formatNum(p.total_tweets, 0))}
            ${statRow("Total Likes",   "Likes on those tweets",                  (parseFloat(p.total_likes)||0).toLocaleString())}
          </div>
        </div>

      </div>
    </div>
  `;
}

// ── Season chart ───────────────────────────────────────────
let chartInstance = null;
let chartData     = {};

function renderChart(labels, goals, assists, minutes, value) {
  const canvas = document.getElementById("playerChart");
  if (!canvas) return;
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }

  chartData = { labels, goals, assists, minutes, value };
  chartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Market Value (€M)",
        data: value,
        borderColor: "#1a7a4a",
        backgroundColor: "rgba(26,122,74,0.1)",
        pointBackgroundColor: "#1a7a4a",
        pointRadius: 5, tension: 0.3, fill: true
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "top", labels: { font: { family: "'Segoe UI', Arial, sans-serif", size: 13 } } },
        tooltip: { callbacks: { label: ctx => " " + ctx.dataset.label + ": " + ctx.parsed.y } }
      },
      scales: {
        x: { grid: { color: "#f0f0f0" }, ticks: { font: { family: "'Segoe UI', Arial, sans-serif" } } },
        y: { beginAtZero: true, grid: { color: "#f0f0f0" }, ticks: { font: { family: "'Segoe UI', Arial, sans-serif" } } }
      }
    }
  });
}

function switchStat(stat, btn) {
  if (!chartInstance || !chartData.labels) return;
  document.querySelectorAll(".chart-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  if (stat === "market_value") {
    chartInstance.data.datasets = [{ label: "Market Value (€M)", data: chartData.value, borderColor: "#1a7a4a", backgroundColor: "rgba(26,122,74,0.1)", pointBackgroundColor: "#1a7a4a", pointRadius: 5, tension: 0.3, fill: true }];
  } else if (stat === "minutes") {
    chartInstance.data.datasets = [{ label: "Minutes Played", data: chartData.minutes, borderColor: "#3b82f6", backgroundColor: "rgba(59,130,246,0.1)", pointBackgroundColor: "#3b82f6", pointRadius: 5, tension: 0.3, fill: true }];
  } else {
    chartInstance.data.datasets = [
      { label: "Goals",   data: chartData.goals,   borderColor: "#1a7a4a", backgroundColor: "rgba(26,122,74,0.1)", pointBackgroundColor: "#1a7a4a", pointRadius: 5, tension: 0.3, fill: true },
      { label: "Assists", data: chartData.assists, borderColor: "#f0b429", backgroundColor: "rgba(240,180,41,0.1)", pointBackgroundColor: "#f0b429", pointRadius: 5, tension: 0.3, fill: true }
    ];
  }
  chartInstance.update();
}


// ── Sentiment donut ────────────────────────────────────────
let sentimentChartInstance = null;

function renderSentimentChart(posPct, negPct, neuPct) {
  const canvas = document.getElementById("sentimentChart");
  if (!canvas) return;
  if (sentimentChartInstance) { sentimentChartInstance.destroy(); sentimentChartInstance = null; }

  sentimentChartInstance = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Positive", "Negative", "Neutral"],
      datasets: [{
        data: [posPct, negPct, neuPct],
        backgroundColor: ["#2ecc71", "#e74c3c", "#bdc3c7"],
        borderColor:     ["#27ae60", "#c0392b", "#95a5a6"],
        borderWidth: 2, hoverOffset: 8
      }]
    },
    options: {
      responsive: true, cutout: "65%",
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => " " + ctx.label + ": " + ctx.parsed + "%" } }
      }
    }
  });
}