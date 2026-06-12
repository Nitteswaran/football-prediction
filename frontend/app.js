/* Pitchsense frontend — talks to the FastAPI backend. */

const API = "";   // same origin

const $ = (sel) => document.querySelector(sel);
const fmtPct = (p, dp = 1) => (100 * p).toFixed(dp) + "%";

/* Extract a human message from a failed response without assuming JSON —
   an edge/proxy (Cloudflare, HF) may return HTML on 5xx/timeouts. */
async function errorDetail(res) {
  const text = await res.text().catch(() => "");
  try {
    return JSON.parse(text).detail || res.statusText;
  } catch {
    return res.statusText || `request failed (${res.status})`;
  }
}

/* ---------------- navigation ---------------- */
document.querySelectorAll(".nav-link").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
    $(`#view-${btn.dataset.view}`).classList.remove("hidden");
    if (btn.dataset.view === "worldcup") loadWorldCup();
    if (btn.dataset.view === "rankings") loadRankings();
  });
});

/* ---------------- teams ---------------- */
function makeCombo(input, teams) {
  const list = document.getElementById(input.getAttribute("aria-controls"));
  let matches = teams;
  let active = -1;
  let lastValid = "";

  const norm = (s) => s.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");

  function filter(q) {
    const nq = norm(q.trim());
    if (!nq) return teams;
    const starts = [], contains = [];
    for (const t of teams) {
      const nt = norm(t);
      if (nt.startsWith(nq)) starts.push(t);
      else if (nt.includes(nq)) contains.push(t);
    }
    return starts.concat(contains);
  }

  function open(q) {
    matches = filter(q);
    active = -1;
    list.innerHTML = "";
    for (const t of matches.slice(0, 200)) {
      const li = document.createElement("li");
      li.setAttribute("role", "option");
      li.textContent = t;
      li.addEventListener("mousedown", (e) => {
        e.preventDefault(); // keep focus on the input
        select(t);
      });
      list.appendChild(li);
    }
    if (!matches.length) {
      const li = document.createElement("li");
      li.className = "combo-empty";
      li.textContent = "No matching team";
      list.appendChild(li);
    }
    list.classList.remove("hidden");
    input.setAttribute("aria-expanded", "true");
  }

  function close() {
    list.classList.add("hidden");
    input.setAttribute("aria-expanded", "false");
    active = -1;
  }

  function select(team) {
    input.value = team;
    lastValid = team;
    close();
  }

  function setActive(i) {
    const items = list.querySelectorAll("[role=option]");
    if (!items.length) return;
    active = (i + items.length) % items.length;
    items.forEach((el, j) => el.classList.toggle("active", j === active));
    items[active].scrollIntoView({ block: "nearest" });
  }

  input.addEventListener("focus", () => { input.select(); open(""); });
  input.addEventListener("input", () => open(input.value));
  input.addEventListener("keydown", (e) => {
    if (list.classList.contains("hidden") && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      open(input.value);
    }
    if (e.key === "ArrowDown") { e.preventDefault(); setActive(active + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(active - 1); }
    else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0) select(matches[active]);
      else if (matches.length) select(matches[0]);
    }
    else if (e.key === "Escape") { close(); }
  });
  input.addEventListener("blur", () => {
    // snap free text to the exact team name (case/accent-insensitive),
    // otherwise restore the last valid pick
    const exact = teams.find((t) => norm(t) === norm(input.value.trim()));
    if (exact) lastValid = exact;
    input.value = lastValid;
    close();
  });

  return { set: select };
}

let homeCombo, awayCombo;

async function loadTeams() {
  const res = await fetch(`${API}/api/teams`);
  const { teams } = await res.json();
  homeCombo = makeCombo($("#home-team"), teams);
  awayCombo = makeCombo($("#away-team"), teams);
  homeCombo.set(teams.includes("Brazil") ? "Brazil" : teams[0]);
  awayCombo.set(teams.includes("Germany") ? "Germany" : teams[1]);
}

/* ---------------- paywall ---------------- */
const UNLOCK_KEY = "ps_unlock_token";
const FIXTURE_KEY = "ps_pending_fixture";

const unlockToken = () => localStorage.getItem(UNLOCK_KEY);

function setLockedUI(locked) {
  $("#result").classList.toggle("locked", locked);
  $("#paywall").classList.toggle("hidden", !locked);
}

/* Plausible decoy numbers to blur behind the paywall — the real insights
   never leave the server until the session is paid. */
function decoyPrediction(home, away) {
  const pois = (lam) => {
    const fact = [1, 1, 2, 6, 24, 120];
    return Array.from({ length: 6 }, (_, k) => Math.exp(-lam) * lam ** k / fact[k]);
  };
  const ph = pois(1.55), pa = pois(1.15);
  const grid = ph.map((a) => pa.map((b) => a * b));
  const scores = [];
  for (let i = 0; i < 6; i++)
    for (let j = 0; j < 6; j++)
      scores.push({ score: `${i}-${j}`, probability: grid[i][j] });
  scores.sort((a, b) => b.probability - a.probability);
  return {
    home_team: home, away_team: away,
    probabilities: { home_win: 0.46, draw: 0.26, away_win: 0.28 },
    expected_goals: { home: 1.55, away: 1.15 },
    top_scorelines: scores.slice(0, 6),
    scoreline_grid: grid,
    drivers: {
      elo_delta: "+112", elo_expectation_home: "65.4%",
      form10_ppg_home: "2.1", form10_ppg_away: "1.4",
      h2h5_weighted_score: "+0.8", rest_delta_days: "+2",
    },
    model_probabilities: {
      xgboost: [0.47, 0.25, 0.28], lightgbm: [0.45, 0.27, 0.28],
      catboost: [0.46, 0.26, 0.28], mlp: [0.44, 0.27, 0.29],
    },
  };
}

$("#unlock-btn").addEventListener("click", async () => {
  const btn = $("#unlock-btn");
  btn.disabled = true;
  btn.textContent = "Redirecting…";
  try {
    localStorage.setItem(FIXTURE_KEY, JSON.stringify({
      home: $("#home-team").value, away: $("#away-team").value,
      neutral: $("#neutral").checked, tournament: $("#tournament").value,
    }));
    const res = await fetch(`${API}/api/checkout`, { method: "POST" });
    if (!res.ok) throw new Error(await errorDetail(res));
    const data = await res.json();
    window.location = data.url;
  } catch (err) {
    alert("Could not start checkout: " + err.message);
    btn.disabled = false;
    btn.textContent = "Unlock for $5";
  }
});

/* Back from Stripe Checkout: verify the session server-side, store the
   unlock token, and re-run the fixture the user was looking at. */
async function handleCheckoutReturn() {
  const params = new URLSearchParams(location.search);
  const sid = params.get("session_id");
  if (params.has("session_id") || params.has("canceled"))
    history.replaceState({}, "", location.pathname);
  if (!sid) return;
  const res = await fetch(`${API}/api/unlock?session_id=${encodeURIComponent(sid)}`);
  const { unlocked } = await res.json();
  if (!unlocked) {
    alert("Payment was not completed — insights remain locked.");
    return;
  }
  localStorage.setItem(UNLOCK_KEY, sid);
  const f = JSON.parse(localStorage.getItem(FIXTURE_KEY) || "null");
  if (f && homeCombo) {
    homeCombo.set(f.home);
    awayCombo.set(f.away);
    $("#neutral").checked = f.neutral;
    $("#tournament").value = f.tournament;
    runPrediction();
  }
  localStorage.removeItem(FIXTURE_KEY);
}

/* ---------------- prediction ---------------- */
$("#predict-btn").addEventListener("click", runPrediction);

async function runPrediction() {
  const btn = $("#predict-btn");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const body = {
      home_team: $("#home-team").value,
      away_team: $("#away-team").value,
      neutral: $("#neutral").checked,
      tournament: $("#tournament").value,
    };
    const headers = { "Content-Type": "application/json" };
    if (unlockToken()) headers["X-Unlock-Token"] = unlockToken();
    const res = await fetch(`${API}/api/predict`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await errorDetail(res));
    const data = await res.json();
    if (data.locked) {
      renderPrediction(decoyPrediction(data.home_team, data.away_team));
      setLockedUI(true);
    } else {
      setLockedUI(false);
      renderPrediction(data);
    }
  } catch (err) {
    alert("Prediction failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run forecast";
  }
}

function renderPrediction(d) {
  $("#result").classList.remove("hidden");
  $("#result-title").textContent = `${d.home_team} v ${d.away_team}`;
  $("#result-tag").textContent = "calibrated ensemble";

  const p = d.probabilities;
  const segs = [
    [".seg-home", p.home_win, `${d.home_team} ${fmtPct(p.home_win)}`],
    [".seg-draw", p.draw, `Draw ${fmtPct(p.draw)}`],
    [".seg-away", p.away_win, `${d.away_team} ${fmtPct(p.away_win)}`],
  ];
  for (const [sel, prob, label] of segs) {
    const el = $(sel);
    el.style.width = (100 * prob) + "%";
    el.querySelector(".seg-label").textContent = prob > 0.12 ? label : fmtPct(prob, 0);
  }
  $("#legend-home").textContent = `${d.home_team} win`;
  $("#legend-away").textContent = `${d.away_team} win`;

  $("#xg-home").textContent = d.expected_goals.home.toFixed(2);
  $("#xg-away").textContent = d.expected_goals.away.toFixed(2);
  $("#xg-home-team").textContent = d.home_team;
  $("#xg-away-team").textContent = d.away_team;

  const list = $("#scoreline-list");
  list.innerHTML = "";
  for (const s of d.top_scorelines.slice(0, 6)) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${s.score.replace("-", " – ")}</span>
                    <span class="pct">${fmtPct(s.probability)}</span>`;
    list.appendChild(li);
  }

  renderHeatmap(d);
  renderDrivers(d.drivers);
  renderModelTable(d.model_probabilities);
}

function renderHeatmap(d) {
  const N = 6; // show 0..5 goals
  const grid = d.scoreline_grid;
  const hm = $("#heatmap");
  hm.innerHTML = "";
  hm.style.gridTemplateColumns = `repeat(${N + 1}, 1fr)`;

  let max = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) max = Math.max(max, grid[i][j]);

  hm.appendChild(cell("", "hm-head"));
  for (let j = 0; j < N; j++) hm.appendChild(cell(String(j), "hm-head"));
  for (let i = 0; i < N; i++) {
    hm.appendChild(cell(String(i), "hm-head"));
    for (let j = 0; j < N; j++) {
      const v = grid[i][j];
      const c = cell(fmtPct(v, 0), "hm-cell");
      const alpha = Math.pow(v / max, 0.7);
      c.style.background = `rgba(11,110,79,${(0.06 + 0.94 * alpha).toFixed(3)})`;
      c.style.color = alpha > 0.45 ? "#fff" : "#161D1A";
      c.title = `${i}–${j}: ${fmtPct(v)}`;
      hm.appendChild(c);
    }
  }
  $("#hm-home").textContent = d.home_team;
  $("#hm-away").textContent = d.away_team;

  function cell(text, cls) {
    const el = document.createElement("div");
    el.className = `hm-cell ${cls}`;
    el.textContent = text;
    return el;
  }
}

const DRIVER_LABELS = {
  elo_home: "Elo — home",
  elo_away: "Elo — away",
  elo_delta: "Elo gap",
  elo_expectation_home: "Elo win expectancy",
  form10_ppg_home: "Form (pts/game, last 10) — home",
  form10_ppg_away: "Form (pts/game, last 10) — away",
  momentum_delta: "Momentum edge",
  h2h5_weighted_score: "Head-to-head (last 5)",
  rest_delta_days: "Rest advantage (days)",
};

function renderDrivers(drivers) {
  const table = $("#drivers-table");
  table.innerHTML = "";
  for (const [key, val] of Object.entries(drivers)) {
    if (val === null || val === undefined) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${DRIVER_LABELS[key] || key}</td><td>${val}</td>`;
    table.appendChild(tr);
  }
}

function renderModelTable(models) {
  const tbody = $("#model-table tbody");
  tbody.innerHTML = "";
  const names = { xgboost: "XGBoost", lightgbm: "LightGBM", catboost: "CatBoost", mlp: "Neural net" };
  for (const [name, probs] of Object.entries(models)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${names[name] || name}</td>
      <td>${fmtPct(probs[0])}</td><td>${fmtPct(probs[1])}</td><td>${fmtPct(probs[2])}</td>`;
    tbody.appendChild(tr);
  }
}

/* ---------------- world cup ---------------- */
let wcLoaded = false;
async function loadWorldCup() {
  if (wcLoaded) return;
  const res = await fetch(`${API}/api/worldcup`);
  if (!res.ok) {
    $("#wc-meta").textContent = "Simulation not generated yet — run `python -m simulation.engine`.";
    return;
  }
  const data = await res.json();
  wcLoaded = true;
  $("#wc-meta").textContent =
    `${data.n_simulations.toLocaleString()} full-tournament simulations · ` +
    `group games sampled from exact-score distributions · official R32 bracket.`;
  const tbody = $("#wc-table tbody");
  tbody.innerHTML = "";
  const maxChamp = data.teams[0].champion || 1;
  data.teams.forEach((t, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="num">${i + 1}</td>
      <td class="team-cell">${t.team}</td>
      <td><span class="group-chip">${t.group}</span></td>
      <td class="num">${fmtPct(t.advance_group)}</td>
      <td class="num">${fmtPct(t.quarterfinal)}</td>
      <td class="num">${fmtPct(t.semifinal)}</td>
      <td class="num">${fmtPct(t.final)}</td>
      <td><div class="champ-bar">
            <div class="bar" style="width:${(100 * t.champion / maxChamp).toFixed(1)}%"></div>
            <span class="val">${fmtPct(t.champion)}</span>
          </div></td>`;
    tbody.appendChild(tr);
  });
}

/* ---------------- rankings ---------------- */
let rankLoaded = false;
async function loadRankings() {
  if (rankLoaded) return;
  const res = await fetch(`${API}/api/rankings?top=80`);
  const { rankings } = await res.json();
  rankLoaded = true;
  const tbody = $("#rank-table tbody");
  tbody.innerHTML = "";
  for (const r of rankings) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="num">${r.rank}</td>
      <td class="team-cell">${r.team}</td>
      <td class="num">${r.elo.toFixed(0)}</td>
      <td class="num">${r.matches}</td>`;
    tbody.appendChild(tr);
  }
}

loadTeams().then(handleCheckoutReturn);
