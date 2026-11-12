/* Stormcast frontend: a single page over the JSON API in src/stormcast/server.py. No build step. */
"use strict";

const C = { amber: "#ffc94d", storm: "#ff5d6c", calm: "#4db4ff", green: "#3ddc97", violet: "#c084fc",
  muted: "#8b96a8", dim: "#5b6678", line: "#222b3a", text: "#e8ecf3" };
const MODEL_COLORS = { har: "#6b778a", gbm_prereg: "#7c8cff", gbm_tuned: C.calm, stormnet: C.violet, ensemble: C.amber };
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const pct = (v, d = 1) => (v == null ? "–" : (100 * v).toFixed(d) + "%");
const api = async (path, opt) => {
  const r = await fetch(path, opt);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const PLOT_CFG = { displayModeBar: false, responsive: true };

function layout(extra = {}) {
  const base = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, system-ui, sans-serif", color: C.muted, size: 12 },
    margin: { l: 48, r: 16, t: 38, b: 34 },
    xaxis: { gridcolor: "#1a2230", zerolinecolor: "#1a2230", linecolor: C.line },
    yaxis: { gridcolor: "#1a2230", zerolinecolor: "#1a2230", linecolor: C.line },
    legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 11.5 } },
    hoverlabel: { bgcolor: "#131924", bordercolor: "#2c3749", font: { family: "Inter", color: C.text } },
    title: { font: { size: 14, color: C.text, family: "Inter" }, x: 0.01, y: 0.97 },
  };
  const out = { ...base, ...extra };
  out.legend = { ...base.legend, ...(extra.legend || {}) };
  if (!extra.margin) out.margin = { ...base.margin, t: extra.title ? 78 : 30 };
  out.xaxis = { ...base.xaxis, ...(extra.xaxis || {}) };
  out.yaxis = { ...base.yaxis, ...(extra.yaxis || {}) };
  if (extra.title && typeof extra.title === "string") out.title = { ...base.title, text: extra.title };
  return out;
}
const plot = (id, data, lay) => Plotly.react(id, data, layout(lay), PLOT_CFG);

const state = { summary: null, day: null, i: 18, playing: false, speed: 1, timer: null, reveal: true, view: "replay",
  sizer: "ensemble", rendered: new Set() };

// ================================================================== navigation & keys
function show(view) {
  state.view = view;
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  if (view !== "replay") pause();
  if (!state.rendered.has(view)) { RENDER[view]?.(); state.rendered.add(view); }
  window.dispatchEvent(new Event("resize"));
  history.replaceState(null, "", `#${view}`);
}
$$("#nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) && e.target.type !== "range") return;
  const views = ["replay", "arena", "trust", "risk", "leak", "forecast", "method"];
  if (e.key >= "1" && e.key <= "7") show(views[+e.key - 1]);
  else if (e.key === "f" || e.key === "F") toggleFs();
  else if (state.view === "replay") {
    if (e.key === " ") { e.preventDefault(); state.playing ? pause() : play(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    else if (e.key === "r" || e.key === "R") setReveal(!state.reveal);
  }
});
function toggleFs() { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
$("#fs").addEventListener("click", toggleFs);

// ================================================================== boot
(async function boot() {
  try {
    state.summary = await api("/api/summary");
  } catch (e) {
    $("#loading").textContent = `Could not load results: ${e.message}. Run "uv run stormcast train" first.`;
    return;
  }
  $("#loading").remove();
  if (state.summary.synthetic) {
    const b = document.createElement("div");
    b.className = "synth";
    b.innerHTML = `<span style="font-size:20px">🧪</span><div><b>Synthetic market.</b> The real E-mini data is licensed, so this
      copy runs on a simulated market with the same statistical structure (volatility clustering, intraday U-shape, regimes,
      FOMC bursts, crashes) and an unpredictable direction. Every model, chart and number here was produced from it. The
      Leak lab's planted-mistake chart is from the real-data study.</div>`;
    $("#app").prepend(b);
    const pill = $(".pill.live"); pill.textContent = "synthetic market"; pill.classList.add("synthetic"); pill.classList.remove("live");
    $("#view-replay .lede").innerHTML = "Machine learning on 3.1 million one-minute bars of a <b>simulated</b> S&amp;P 500 futures market. It forecasts <b>how much</b> the market will move in the next 30 minutes, and shows honestly that it cannot forecast <b>which way</b>.";
  }
  renderKpis();
  setupReplay();
  const start = (location.hash || "#replay").slice(1);
  show(RENDER[start] ? start : "replay");
})();

function renderKpis() {
  const k = state.summary.kpi;
  const cards = [
    ["Size of the next move: R²", fmt(k.r2, 2), `+${fmt(k.r2 - k.r2_har, 3)} vs HAR econometrics`, "good"],
    ["Direction of the next move: AUC", fmt(k.dir_auc, 3), "a coin flip scores 0.500", "warn"],
    ["90% interval coverage", pct(k.coverage), "adaptive conformal, target 90%", ""],
    ["Forecasts scored", k.n.toLocaleString(), `${k.days.toLocaleString()} trading days, walk-forward`, ""],
    ["Stacked models", `3 <span style="font-size:15px;color:var(--muted);font-weight:600">+ ${k.seeds}-seed deep ensemble</span>`, "HAR · GBM + Optuna · StormNet", ""],
  ];
  $("#kpis").innerHTML = cards.map(([l, v, s, t]) =>
    `<div class="card kpi"><div class="l">${l}</div><div class="v ${t}">${v}</div><div class="s ${t}">${s}</div></div>`).join("");
  $("#m-r2").textContent = `${Math.round(100 * k.r2)}%`;
}

// ================================================================== replay
function setupReplay() {
  const s = state.summary;
  $("#presets").innerHTML = s.presets.map((p) =>
    `<button class="chip" data-date="${p.date}" ${p.available ? "" : "disabled"} title="${p.note}">${p.name}</button>`).join("");
  $$("#presets .chip").forEach((c) => c.addEventListener("click", () => loadDay(c.dataset.date)));
  const dp = $("#datepick");
  dp.min = s.dates[0]; dp.max = s.dates[s.dates.length - 1];
  dp.addEventListener("change", () => {
    const want = dp.value, i = s.dates.findIndex((d) => d >= want);
    loadDay(s.dates[i < 0 ? s.dates.length - 1 : i]);
  });
  $("#slider").addEventListener("input", (e) => { pause(); setSlot(+e.target.value); });
  $("#play").addEventListener("click", () => (state.playing ? pause() : play()));
  $("#next").addEventListener("click", () => { pause(); step(1); });
  $("#prev").addEventListener("click", () => { pause(); step(-1); });
  $$("#speed button").forEach((b) => b.addEventListener("click", () => {
    state.speed = +b.dataset.s; $$("#speed button").forEach((x) => x.classList.toggle("on", x === b));
    if (state.playing) { pause(); play(); }
  }));
  setupWhatIf();
  const first = s.presets.find((p) => p.available) || { date: s.dates[s.dates.length - 1] };
  loadDay(first.date);
}

async function loadDay(date) {
  pause();
  $$("#presets .chip").forEach((c) => c.classList.toggle("active", c.dataset.date === date));
  $("#datepick").value = date;
  const d = await api(`/api/day/${date}`);
  d.barT = d.bars.t.map((t) => new Date(t + ":00"));
  state.day = d;
  const sl = $("#slider");
  sl.max = d.slots.length - 1;
  const preset = state.summary.presets.find((p) => p.date === date);
  $("#daybar").innerHTML = `<span class="title">${d.weekday} ${new Date(date + "T12:00").toLocaleDateString("en-US",
    { day: "numeric", month: "long", year: "numeric" })}</span>
    ${preset ? `<span class="tag">${preset.name}: ${preset.note}</span>` : ""}
    <span class="tag">models trained on data before ${d.year}</span>
    <span class="tag ${d.post_hoc ? "post" : ""}">${d.post_hoc ? "post hoc year" : "development year"}</span>
    <span class="tag">day: ${pct(d.day_stats.coverage, 0)} of intervals hit · forecast/actual corr ${fmt(d.day_stats.corr, 2)}</span>`;
  setSlot(Math.min(state.i, d.slots.length - 1), true);
}

function step(n) {
  const d = state.day; if (!d) return;
  setSlot(Math.max(0, Math.min(d.slots.length - 1, state.i + n)));
}
function play() {
  const d = state.day; if (!d) return;
  if (state.i >= d.slots.length - 1) setSlot(0);
  state.playing = true; $("#play").textContent = "❚❚";
  state.timer = setInterval(() => {
    if (state.i >= state.day.slots.length - 1) return pause();
    step(1);
  }, 650 / state.speed);
}
function pause() { state.playing = false; clearInterval(state.timer); $("#play").textContent = "▶"; }
function setReveal(v) { state.reveal = v; renderSlot(); }

function setSlot(i, newDay = false) {
  state.i = i;
  const sl = $("#slider");
  sl.value = i;
  sl.style.setProperty("--p", `${(100 * i) / Math.max(1, sl.max)}%`);
  renderSlot(newDay);
}

function slotTime(d, s) { return new Date(`${d.date}T${s.t}:00`); }
function idxAtOrAfter(d, t) { // first bar at/after t (bars are sorted)
  let lo = 0, hi = d.barT.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (d.barT[m] < t) lo = m + 1; else hi = m; }
  return lo;
}

function condition(s) {
  const r = s.vol / s.vol_usual;
  if (r > 1.6 || s.vol > 30) return { icon: "⛈", name: "Storm warning", cls: "bad" };
  if (r > 1.15) return { icon: "🌧", name: "Unsettled", cls: "warn" };
  if (r < 0.85) return { icon: "☀️", name: "Calm", cls: "good" };
  return { icon: "⛅", name: "Normal", cls: "" };
}

function renderSlot() {
  const d = state.day; if (!d) return;
  const s = d.slots[state.i];
  $("#clock").innerHTML = `${s.t}<small>ET</small>`;
  const t0 = slotTime(d, s), t30 = new Date(+t0 + 30 * 60000);
  const i0 = idxAtOrAfter(d, t0), i30 = idxAtOrAfter(d, t30);
  const rthStart = d.bars.rth.indexOf(true);
  const sliceT = (a, b) => d.barT.slice(a, b), sliceC = (a, b) => d.bars.c.slice(a, b);

  // ---- price chart with forecast cone
  const traces = [
    { x: sliceT(rthStart, i0), y: sliceC(rthStart, i0), mode: "lines", line: { color: C.text, width: 1.7 }, name: "price",
      hovertemplate: "%{x|%H:%M}  %{y:.2f}<extra></extra>" },
  ];
  const N = 31, tau = [...Array(N).keys()];
  const xs = tau.map((m) => new Date(+t0 + m * 60000));
  const sdUp = tau.map((m) => s.price + s.sd_pts * Math.sqrt(m / 30)), sdDn = tau.map((m) => s.price - s.sd_pts * Math.sqrt(m / 30));
  traces.push({ x: [t0, t30, t30, t0, t0], y: [s.price + s.range_hi, s.price + s.range_hi, s.price - s.range_hi,
    s.price - s.range_hi, s.price + s.range_hi], fill: "toself", fillcolor: "rgba(255,93,108,.10)", mode: "lines",
    line: { color: "rgba(255,93,108,.45)", width: 1, dash: "dot" }, name: "90% envelope (whole 30 min)", hoverinfo: "skip" });
  traces.push({ x: [...xs, ...xs.slice().reverse()], y: [...sdUp, ...sdDn.slice().reverse()], fill: "toself",
    fillcolor: "rgba(255,201,77,.20)", line: { width: 0 }, mode: "lines", name: "±1σ cone", hoverinfo: "skip" });
  if (state.reveal) {
    traces.push({ x: sliceT(i0 - 1, i30), y: sliceC(i0 - 1, i30), mode: "lines", line: { color: C.amber, width: 2.2 },
      name: "next 30 min", hovertemplate: "%{x|%H:%M}  %{y:.2f}<extra></extra>" });
    traces.push({ x: sliceT(i30 - 1, d.barT.length), y: sliceC(i30 - 1, d.barT.length), mode: "lines",
      line: { color: "#3a4558", width: 1.2 }, name: "later", hoverinfo: "skip" });
  }
  traces.push({ x: [t0], y: [s.price], mode: "markers", marker: { color: C.amber, size: 9, line: { color: "#000", width: 2 } },
    showlegend: false, hoverinfo: "skip" });
  const rthC = d.bars.c.slice(rthStart);
  const yr = [Math.min(...rthC), Math.max(...rthC)], pad = (yr[1] - yr[0]) * 0.08 + 1;
  $("#price-title").innerHTML = `ES futures · forecast cone at ${s.t} ET <span class="cap">yellow: ±1σ (√t) · red: 90% envelope for the whole 30 minutes</span>`;
  plot("chart-price", traces, {
    showlegend: true, margin: { l: 56, r: 16, t: 34, b: 30 },
    xaxis: { range: [d.barT[rthStart], d.barT[d.barT.length - 1]], tickformat: "%H:%M" },
    yaxis: { range: [Math.min(yr[0], s.price - s.range_hi) - pad, Math.max(yr[1], s.price + s.range_hi) + pad], title: { text: "index points" } },
    shapes: [{ type: "line", x0: t0, x1: t0, yref: "paper", y0: 0, y1: 1, line: { color: C.amber, width: 1, dash: "dot" } }],
  });

  // ---- volatility through the day
  const known = d.slots.filter((x, j) => j <= state.i);
  const revealT = new Date(+t0 - (state.reveal ? 0 : 30 * 60000));
  const revealed = d.slots.filter((x) => slotTime(d, x) <= revealT);
  const allT = d.slots.map((x) => slotTime(d, x));
  plot("chart-vol", [
    { x: allT, y: d.slots.map((x) => x.vol_usual), name: "usual for this time of day", mode: "lines",
      line: { color: C.dim, dash: "dot", width: 1.5 } },
    { x: known.map((x) => slotTime(d, x)), y: known.map((x) => x.vol), name: "Stormcast forecast", mode: "lines",
      line: { color: C.amber, width: 2.6, shape: "spline" } },
    { x: revealed.map((x) => slotTime(d, x)), y: revealed.map((x) => x.vol_actual), name: "actual (once known)", mode: "markers",
      marker: { color: C.storm, size: 6 } },
  ], { xaxis: { range: [allT[0], allT[allT.length - 1]], tickformat: "%H:%M" }, 
    yaxis: { rangemode: "tozero" }, margin: { l: 48, r: 16, t: 30, b: 30 },
    shapes: [{ type: "line", x0: t0, x1: t0, yref: "paper", y0: 0, y1: 1, line: { color: C.amber, width: 1, dash: "dot" } }] });

  renderReport(d, s);
  renderAttention(d, s, i0);
  syncWhatIf(s);
}

function renderReport(d, s) {
  const c = condition(s);
  const diff = s.vol - s.vol_usual;
  const maxV = Math.max(60, s.vol * 1.1);
  const sh = Object.entries(s.shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 7);
  const shMax = Math.max(0.05, ...sh.map(([, v]) => Math.abs(v)));
  const labels = Object.fromEntries(state.summary.features.map((f) => [f.key, f.label]));
  const reg = d.regime;
  const regCell = (n, p, col) => `<div style="${p > 0.5 ? `border-color:${col};box-shadow:0 0 18px ${col}33` : ""}">
    <div class="n">${n}</div><div class="p" style="color:${p > 0.5 ? col : C.muted}">${pct(p, 0)}</div></div>`;
  $("#report").innerHTML = `
    <div class="wx">
      <div class="icon">${c.icon}</div>
      <div>
        <div class="cond ${c.cls}">${c.name}</div>
        <div class="big">${fmt(s.vol, 0)}%<small> annualized vol</small></div>
        <div class="vs"><span class="${diff > 3 ? "bad" : diff < -3 ? "good" : ""}">${diff >= 0 ? "+" : ""}${fmt(diff, 0)} pts</span> vs usual at ${s.t} · VIX ${fmt(s.vix, 1)}</div>
      </div>
    </div>
    <div class="meter"><i style="width:${(100 * s.vol) / maxV}%; background:linear-gradient(90deg, ${C.calm}, ${s.vol / s.vol_usual > 1.3 ? C.storm : C.amber})"></i></div>
    <div class="stat3">
      <div><div class="k">±1σ move</div><div class="val">${fmt(s.sd_pts, 1)}<span style="font-size:12px;color:var(--muted)"> pts</span></div></div>
      <div><div class="k">90% range</div><div class="val">${fmt(s.range_lo, 1)}–${fmt(s.range_hi, 1)}</div></div>
      <div><div class="k">Position size</div><div class="val">${fmt(s.size, 2)}×</div></div>
    </div>
    <div class="coin">
      <div class="face">½</div>
      <div style="flex:1">
        <div class="row" style="margin:0"><span class="k">P(up in 30 min)</span><span class="val warn">${fmt(s.p_up, 2)}</span></div>
        <div class="cap" style="margin:2px 0 0">A coin flip, honestly reported. Never trade the direction.</div>
      </div>
    </div>
    ${reg ? `<div><h3 style="margin-bottom:8px">Market regime <span class="cap">hidden Markov model, known at the open</span></h3>
      <div class="regime">${regCell("calm", reg.calm, C.calm)}${regCell("normal", reg.normal, C.muted)}${regCell("storm", reg.storm, C.storm)}</div></div>` : ""}
    <div><h3 style="margin-bottom:6px">Why? Top drivers <span class="cap">exact SHAP, gradient boosting</span></h3>
      <div class="bars">${sh.map(([k, v]) => {
        const w = (50 * Math.abs(v)) / shMax;
        return `<div class="b"><span class="lab" title="${labels[k]}">${labels[k]}</span><span class="track"><span class="mid"></span>
          <i style="left:${v >= 0 ? 50 : 50 - w}%; width:${w}%; background:${v >= 0 ? C.storm : C.calm}"></i></span>
          <span class="num">${v >= 0 ? "+" : ""}${fmt(v, 2)}</span></div>`;
      }).join("")}</div>
      <div class="cap" style="margin-top:4px">← calmer · stormier →  Members: HAR ${fmt(s.members.har, 0)}% · GBM ${fmt(s.members.gbm_tuned, 0)}% · StormNet ${fmt(s.members.stormnet, 0)}%</div>
    </div>
    <div class="outcome ${state.reveal ? "" : "hidden"}">
      <div class="hd"><span>30 minutes later…</span><span class="toggle" id="reveal-t"><span class="switch ${state.reveal ? "on" : ""}"></span>reveal (R)</span></div>
      <div class="body">Actual volatility <b>${fmt(s.vol_actual, 0)}%</b> · range <b>${fmt(s.range_actual, 1)} pts</b>
        <b class="${s.covered ? "good" : "bad"}">${s.covered ? "✓ inside" : "✗ outside"}</b> the 90% interval ·
        price went <b>${s.went_up ? "up" : "down"}</b> ${fmt(Math.abs(s.move_pts), 2)} pts</div>
    </div>`;
  $("#reveal-t").addEventListener("click", () => setReveal(!state.reveal));
}

function renderAttention(d, s, i0) {
  const start = Math.max(0, i0 - 240);
  const winT = d.barT.slice(start, i0), winC = d.bars.c.slice(start, i0);
  const tokT = s.attn.map((_, j) => winT[Math.min(winT.length - 1, j * 4)]);
  plot("chart-attn", [
    { x: tokT, y: s.attn, type: "bar", name: "attention weight", marker: { color: "rgba(192,132,252,.6)" }, yaxis: "y",
      hovertemplate: "%{x|%H:%M}  weight %{y:.3f}<extra></extra>" },
    { x: winT, y: winC, mode: "lines", name: "price", line: { color: C.text, width: 1.3 }, yaxis: "y2", hoverinfo: "skip" },
  ], { margin: { l: 44, r: 50, t: 30, b: 30 }, showlegend: true, bargap: 0.05,
    xaxis: { tickformat: "%H:%M" }, yaxis: { title: { text: "weight" } },
    yaxis2: { overlaying: "y", side: "right", gridcolor: "rgba(0,0,0,0)", color: C.muted } });
}

// ---- what-if
let wiTimer = null;
function setupWhatIf() {
  ["wi-vix", "wi-mult", "wi-k"].forEach((id) => $(`#${id}`).addEventListener("input", whatIfChanged));
  $("#wi-fomc").addEventListener("click", () => { $("#wi-fomc").dataset.on = $("#wi-fomc").dataset.on === "1" ? "0" : "1"; whatIfChanged(); });
  $("#wi-reset").addEventListener("click", () => { state.wiSlot = null; syncWhatIf(state.day.slots[state.i], true); });
}
function syncWhatIf(s, force = false) {
  if (!force && state.wiSlot === `${state.day.date}-${s.k}`) return;
  state.wiSlot = `${state.day.date}-${s.k}`;
  $("#wi-vix").value = s.vix; $("#wi-mult").value = 0; $("#wi-k").value = state.i;
  $("#wi-fomc").dataset.on = s.fomc ? "1" : "0";
  whatIfChanged(true);
}
function whatIfChanged(immediate) {
  const d = state.day, s = d.slots[state.i];
  const vix = +$("#wi-vix").value, m = 2 ** +$("#wi-mult").value, kNew = d.slots[+$("#wi-k").value];
  const on = $("#wi-fomc").dataset.on === "1";
  $("#wi-vix-v").textContent = fmt(vix, 1);
  $("#wi-mult").style.setProperty("--p", `${((+$("#wi-mult").value + 2) / 4) * 100}%`);
  $("#wi-vix").style.setProperty("--p", `${((vix - 9) / 41) * 100}%`);
  $("#wi-k").style.setProperty("--p", `${(100 * +$("#wi-k").value) / 66}%`);
  $("#wi-mult-v").textContent = `×${m < 1 ? fmt(m, 2) : fmt(m, m % 1 ? 2 : 0)}`;
  $("#wi-k-v").textContent = kNew.t;
  $("#wi-fomc .switch").classList.toggle("on", on); $("#wi-fomc span:last-child").textContent = on ? "on" : "off";
  clearTimeout(wiTimer);
  wiTimer = setTimeout(async () => {
    try {
      const r = await api("/api/whatif", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date: d.date, k: s.k, vix, vol_mult: m, fomc: on, k_new: kNew.k }) });
      const diff = r.new - r.base;
      $("#wi-new").innerHTML = `${fmt(r.new, 0)}% <span style="font-size:15px" class="${diff > 0.5 ? "bad" : diff < -0.5 ? "good" : ""}">${diff >= 0 ? "+" : ""}${fmt(diff, 1)}</span>`;
      $("#wi-base").textContent = `${fmt(r.base, 0)}%`;
      const mx = Math.max(0.05, ...r.drivers.map((x) => Math.abs(x.value)));
      $("#wi-drivers").innerHTML = r.drivers.slice(0, 4).map((x) => {
        const w = (50 * Math.abs(x.value)) / mx;
        return `<div class="b"><span class="lab">${x.label}</span><span class="track"><span class="mid"></span>
          <i style="left:${x.value >= 0 ? 50 : 50 - w}%; width:${w}%; background:${x.value >= 0 ? C.storm : C.calm}"></i></span>
          <span class="num">${x.value >= 0 ? "+" : ""}${fmt(x.value, 2)}</span></div>`;
      }).join("");
    } catch (e) { $("#wi-new").textContent = "–"; }
  }, immediate === true ? 0 : 120);
}

// ================================================================== other views
const RENDER = {
  replay() {},
  arena() {
    const s = state.summary;
    const best = s.leaderboard.reduce((a, b) => ((b.r2_all ?? -1) > (a.r2_all ?? -1) ? b : a));
    const cell = (v, d = 3) => `<td class="num">${v == null ? "–" : fmt(v, d)}</td>`;
    $("#board").innerHTML = `<tr><th>Model</th><th class="num">R² all years</th><th class="num">R² dev 2012–15</th>
      <th class="num">R² 2016–19 (post hoc)</th><th class="num">QLIKE ↓</th><th class="num">Diebold–Mariano t vs HAR</th></tr>` +
      s.leaderboard.filter((r) => r.r2_all != null).map((r) => `<tr class="${r.model === best.model ? "best" : ""}"><td><span class="swatch" style="background:${MODEL_COLORS[r.model]}"></span>${r.name}</td>
        ${cell(r.r2_all)}${cell(r.r2_dev)}${cell(r.r2_post)}${cell(r.qlike)}${r.dm_t == null ? '<td class="num">baseline</td>' : cell(r.dm_t, 1)}</tr>`).join("");
    const yrs = s.years.map(String);
    plot("chart-years", Object.keys(MODEL_COLORS).filter((m) => s.r2_by_year[m].some((v) => v != null)).map((m) => ({ x: yrs, y: s.r2_by_year[m], type: "bar", name: s.model_names[m],
      marker: { color: MODEL_COLORS[m] } })), { title: "R² by year: log realized variance, next 30 minutes", barmode: "group",
      yaxis: { range: [0.3, 0.9] } });
    const W = s.weights, ys = Object.keys(W);
    plot("chart-weights", ["har", "gbm_tuned", "stormnet"].map((m) => ({ x: ys, y: ys.map((y) => W[y][m]), type: "bar",
      name: s.model_names[m], marker: { color: MODEL_COLORS[m] } })), { title: "Stacking weights, refit each year on earlier out-of-sample forecasts",
      barmode: "stack", xaxis: { type: "category" } });
    plot("chart-tuning", Object.entries(s.tuning).map(([y, v]) => ({ y: v.best, mode: "lines", name: y, line: { width: 1.8 } })),
      { title: "Optuna: best inner-validation MSE ÷ pre-registered setting", yaxis: { tickformat: ".1%" }, xaxis: { title: { text: "trial" } },
        margin: { l: 54, r: 10, t: 78, b: 40 } });
    const tr = [];
    Object.entries(s.training).forEach(([y, hs], yi) => hs.forEach((h, j) => tr.push({ y: h, mode: "lines", name: y, legendgroup: y,
      showlegend: j === 0, line: { width: 1.2, color: `hsl(${(yi * 45) % 360},70%,65%)` } })));
    plot("chart-training", tr, { title: "StormNet validation loss per seed (early stopping)", xaxis: { title: { text: "epoch" } },
      margin: { l: 44, r: 10, t: 78, b: 40 } });
  },
  trust() {
    const s = state.summary;
    plot("chart-coverage", [
      { x: s.coverage.t, y: s.coverage.raw, name: "raw network quantiles", mode: "lines", line: { color: C.dim, width: 1.2 } },
      { x: s.coverage.t, y: s.coverage.aci, name: "+ adaptive conformal (ACI)", mode: "lines", line: { color: C.amber, width: 1.8 } },
    ], { title: "Rolling coverage of 90% range intervals (last 250 forecasts)", yaxis: { range: [0.4, 1], tickformat: ".0%" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.9, y1: 0.9, line: { color: C.storm, dash: "dot" } }] });
    const cal = s.calibration, lim = [Math.min(...cal.pred, ...cal.rmse) * 0.9, Math.max(...cal.pred, ...cal.rmse) * 1.08];
    plot("chart-cal", [
      { x: lim, y: lim, mode: "lines", name: "perfectly calibrated", line: { color: C.dim, dash: "dot" } },
      { x: cal.pred, y: cal.rmse, mode: "lines+markers", name: "StormNet", line: { color: C.violet, width: 2 }, marker: { size: 9 } },
    ], { title: "Does the model know when it doesn't know?", xaxis: { title: { text: "predicted sd (aleatoric + epistemic), deciles" } },
      yaxis: { title: { text: "actual error (RMSE)" } }, margin: { l: 54, r: 10, t: 78, b: 44 } });
    $("#cal-cap").textContent = `${pct(s.kpi.self_cal)} of outcomes fall inside the model's own 90% uncertainty band (target 90%). ` +
      "Aleatoric = noise the model expects; epistemic = disagreement between the ensemble's seeds.";
    const covBars = (id, o, title) => plot(id, [
      { x: o.labels, y: o.raw, type: "bar", name: "raw quantiles", marker: { color: "#3a4558" } },
      { x: o.labels, y: o.aci, type: "bar", name: "+ ACI", marker: { color: C.amber } },
    ], { title, barmode: "group", yaxis: { range: [0.5, 1], tickformat: ".0%" },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.9, y1: 0.9, line: { color: C.storm, dash: "dot" } }] });
    covBars("chart-covvix", s.cov_vix, "Coverage by VIX level");
    covBars("chart-covyear", s.cov_year, "Coverage by year");
  },
  risk() {
    $("#lim").addEventListener("input", renderRisk);
    $$("#sizer button").forEach((b) => b.addEventListener("click", () => {
      state.sizer = b.dataset.m; $$("#sizer button").forEach((x) => x.classList.toggle("on", x === b)); renderRisk();
    }));
    renderRisk();
  },
  leak() {
    const s = state.summary, L = s.leak.filter((r) => r.metric === "AUC");
    plot("chart-leaks", [
      { y: L.map((r) => r.leak), x: L.map((r) => r.honest), type: "bar", orientation: "h", name: "honest pipeline", marker: { color: C.calm } },
      { y: L.map((r) => r.leak), x: L.map((r) => r.leaky), type: "bar", orientation: "h", name: "with the mistake", marker: { color: C.storm } },
    ], { title: "Direction AUC: honest vs leaky", barmode: "group", xaxis: { range: [0.45, 1] }, yaxis: { autorange: "reversed" },
      margin: { l: 150, r: 10, t: 78, b: 30 },
      shapes: [{ type: "line", x0: 0.5, x1: 0.5, yref: "paper", y0: 0, y1: 1, line: { color: C.dim, dash: "dot" } }] });
    const dd = s.direction;
    plot("chart-dir", [
      { x: dd.days, y: dd.gross, name: "gross points, trading StormNet's direction head", mode: "lines", line: { color: C.amber } },
      { x: dd.days, y: dd.net, name: "after 0.70-point round-trip costs", mode: "lines", line: { color: C.storm } },
    ], { title: "And the honest direction model?", yaxis: { title: { text: "cumulative index points" } }, margin: { l: 60, r: 10, t: 78, b: 30 } });
    const [lo, hi] = s.kpi.dir_auc_ci;
    $("#dir-cap").textContent = `Direction AUC ${fmt(s.kpi.dir_auc, 3)} (95% CI ${fmt(lo, 3)}–${fmt(hi, 3)}): statistically faint at best, economically zero.`;
    $("#leak-run").addEventListener("click", runLeak);
  },
  forecast() {
    const s = state.summary, sel = $("#fc-time");
    for (let k = 30; k <= 360; k += 5) {
      const m = 570 + k, o = document.createElement("option");
      o.value = o.textContent = `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
      sel.appendChild(o);
    }
    $("#fc-date").min = s.dates[0]; $("#fc-date").max = s.dates[s.dates.length - 1];
    $("#fc-date").value = "2018-02-05"; sel.value = "15:00";
    const ex = s.synthetic
      ? s.presets.slice(0, 4).map((p, i) => [p.date, ["14:30", "13:00", "11:00", "14:05"][i], p.name])
      : [["2018-02-05", "15:00", "Volmageddon, 3 pm"], ["2016-06-24", "10:00", "Brexit, open"],
        ["2015-08-24", "10:00", "Aug 2015, open"], ["2017-07-14", "12:30", "Calm lunchtime"]];
    $("#fc-date").value = ex[0][0]; sel.value = ex[0][1];
    $("#fc-examples").innerHTML = ex.map(([d, t, n]) => `<button class="chip" data-d="${d}" data-t="${t}">${n}</button>`).join("");
    $$("#fc-examples .chip").forEach((c) => c.addEventListener("click", () => {
      $("#fc-date").value = c.dataset.d; sel.value = c.dataset.t; runForecast();
    }));
    $("#fc-go").addEventListener("click", runForecast);
    runForecast();
  },
  method() {},
};

function renderRisk() {
  const R = state.summary.risk, mult = +$("#lim").value, f = R[state.sizer], c = R.constant, o = R.oracle;
  $("#lim-v").textContent = `${fmt(mult, 2)}×`;
  $("#lim").style.setProperty("--p", `${((mult - 1.5) / 3.5) * 100}%`);
  const limit = mult * (c.reduce((a, v) => a + Math.abs(v), 0) / c.length);
  const bc = c.filter((v) => v < -limit).length, bf = f.filter((v) => v < -limit).length;
  $("#br-c").textContent = `${bc} days`; $("#br-c-s").textContent = `${pct(bc / c.length)} of ${c.length.toLocaleString()} trading days`;
  $("#br-f").textContent = `${bf} days`;
  $("#br-f-s").textContent = bc ? `${Math.round((100 * (bf - bc)) / bc)}% vs constant size` : "";
  $("#worst").textContent = `${fmt(Math.min(...c), 1)} → ${fmt(Math.min(...f), 1)}`;
  const roll = (a, w = 60) => a.map((_, i) => {
    if (i < w - 1) return null;
    const s = a.slice(i - w + 1, i + 1), m = s.reduce((x, y) => x + y, 0) / w;
    return Math.sqrt(s.reduce((x, y) => x + (y - m) ** 2, 0) / (w - 1));
  });
  plot("chart-rollrisk", [
    { x: R.days, y: roll(c), name: "constant size", mode: "lines", line: { color: "#6b778a", width: 1.8 } },
    { x: R.days, y: roll(f), name: "forecast-sized", mode: "lines", line: { color: C.amber, width: 2.2 } },
    { x: R.days, y: roll(o), name: "oracle (knows the future)", mode: "lines", line: { color: C.calm, width: 1, dash: "dot" } },
  ], { title: "Risk you are actually running: 60-day standard deviation of daily P&L (points)" });
  const idx = c.map((v, i) => i).sort((a, b) => c[a] - c[b]).slice(0, 40).sort((a, b) => a - b);
  plot("chart-worst", [
    { x: idx.map((i) => R.days[i]), y: idx.map((i) => c[i]), type: "bar", name: "constant size", marker: { color: "#4a5568" } },
    { x: idx.map((i) => R.days[i]), y: idx.map((i) => f[i]), type: "bar", name: "forecast-sized", marker: { color: C.amber } },
  ], { title: "The 40 worst days for constant sizing, and the same days forecast-sized", barmode: "group",
    xaxis: { type: "category", tickangle: -55, tickfont: { size: 10 } }, margin: { l: 48, r: 16, t: 78, b: 70 },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: -limit, y1: -limit, line: { color: C.storm, dash: "dash" } }],
    annotations: [{ xref: "paper", x: 1, y: -limit, text: "daily loss limit", showarrow: false, yanchor: "bottom", xanchor: "right",
      font: { color: C.storm, size: 11 } }] });
}

function countUp(el, to, ms = 900) {
  const t0 = performance.now(), from = 0.5;
  const tick = (t) => { const p = Math.min(1, (t - t0) / ms), e = 1 - (1 - p) ** 3; el.textContent = (from + (to - from) * e).toFixed(3);
    if (p < 1) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
}
async function runLeak() {
  const b = $("#leak-run");
  b.disabled = true; b.innerHTML = '<span class="spinner"></span>Training two random forests…';
  try {
    const r = await api("/api/leak", { method: "POST" });
    countUp($("#leak-shuf"), r.shuffled_auc); countUp($("#leak-wf"), r.walk_forward_auc);
    $("#leak-shuf").className = "big bad"; $("#leak-wf").className = "big good";
    $("#leak-shuf-s").innerHTML = "AUC · <b class='bad'>looks like a money machine</b>";
    $("#leak-wf-s").innerHTML = "AUC · <b class='good'>the truth: a coin flip</b>";
    b.textContent = "Run again";
  } catch (e) { b.textContent = `Failed: ${e.message}`; }
  b.disabled = false;
}

async function runForecast() {
  const d = $("#fc-date").value, t = $("#fc-time").value, out = $("#fc-out");
  out.innerHTML = `<span class="p">$</span> stormcast forecast "${d} ${t}"\n\n<span class="c">rebuilding features from bars before ${t}, loading the ${d.slice(0, 4)} models…</span>`;
  try {
    const f = await api(`/api/forecast?when=${encodeURIComponent(`${d} ${t}`)}`);
    const mx = Math.max(...f.drivers.map((x) => Math.abs(x.value)), 1e-6);
    const bar = (v) => "█".repeat(Math.max(1, Math.round((12 * Math.abs(v)) / mx)));
    out.innerHTML = `<span class="p">$</span> stormcast forecast "${d} ${t}"

<span class="a">Stormcast forecast for ${f.time} ET</span>   (models trained on data before ${f.model_year})

  ES price                ${f.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}
  next-30-min volatility  <span class="a">${fmt(f.vol, 1)}%</span> annualized   (1σ move ≈ ${fmt(f.sd_pts, 2)} pts)
  90% high-low range      ${fmt(f.range_lo, 2)} to ${fmt(f.range_hi, 2)} pts  (median ${fmt(f.range_med, 2)})
  90% price envelope      ${fmt(f.price - f.range_hi, 2)} to ${fmt(f.price + f.range_hi, 2)}
  P(up in 30 min)         ${fmt(f.p_up, 3)}   <span class="r">(a coin flip; do not trade it)</span>
  uncertainty             aleatoric sd ${fmt(f.aleatoric, 2)} · epistemic sd ${fmt(f.epistemic, 3)}
  members                 HAR ${fmt(f.members.har, 1)}% · GBM ${fmt(f.members.gbm_tuned, 1)}% · StormNet ${fmt(f.members.stormnet, 1)}%

  top drivers (SHAP):
${f.drivers.map((x) => `    <span class="${x.value >= 0 ? "r" : "c"}">${(x.value >= 0 ? "+" : "-") + fmt(Math.abs(x.value), 3)}  ${bar(x.value).padEnd(13)}</span> ${x.label}`).join("\n")}`;
  } catch (e) { out.innerHTML += `\n\n<span class="r">error: ${e.message}</span>`; }
}
