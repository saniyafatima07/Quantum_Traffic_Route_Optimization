"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  data: null,
  selected: null,
  frame: 0,
  playing: true,
  lastTick: 0,
  convergence: null,
  projection: null,
  legs: 0,
};

const VEHICLE_COLOURS = [
  "#4cc9f0", "#f4a261", "#b388ff", "#6ee7a8", "#f07178",
  "#ffd166", "#7bdff2", "#ef8354", "#a0c4ff", "#95d5b2",
];

const FAMILY_LABEL = {
  quantum: "quantum",
  "quantum-hybrid": "quantum + LS",
  classical: "classical",
};

/* ------------------------------------------------------------------ polling */

async function boot() {
  for (;;) {
    const snapshot = await fetch("/api/state").then((r) => r.json());
    if (snapshot.ready) {
      $("progress").hidden = true;
      $("app").hidden = false;
      await loadData();
      return;
    }
    if (snapshot.status === "failed") {
      $("stage").textContent = "the comparison failed to build";
      $("error").textContent = snapshot.stage || snapshot.error || "";
      return;
    }
    $("fill").style.width = `${Math.round(snapshot.fraction * 100)}%`;
    $("stage").textContent = snapshot.stage || "building…";
    await new Promise((r) => setTimeout(r, 700));
  }
}

async function loadData() {
  const response = await fetch("/api/data");
  if (!response.ok) {
    $("progress").hidden = false;
    $("stage").textContent = "waiting for data";
    return setTimeout(loadData, 800);
  }
  state.data = await response.json();
  renderMeta();
  renderCards();
  renderTable();
  renderConvergence();
  renderHeadToHead();
  buildProjection();
  // Open on the best method, not the first in the list: the page exists to show
  // what the quantum-inspired search found, and burying that behind a click
  // makes the demonstration weaker than the result it is demonstrating.
  const best = rankedRuns()[0];
  select(best ? best.key : (state.data.runs[0] || {}).key || null);
  requestAnimationFrame(tick);
}

$("rebuild").addEventListener("click", async () => {
  $("app").hidden = true;
  $("progress").hidden = false;
  $("error").textContent = "";
  // The fill is only ever written by the poll loop, which leaves it wherever the
  // previous build finished.  Without this a rebuild starts at 100% and sits
  // there, so the bar reports completion for a build that has not begun.
  $("fill").style.width = "0%";
  $("stage").textContent = "starting…";
  await fetch("/api/rebuild", { method: "POST" });
  boot();
});

/* -------------------------------------------------------------------- header */

function renderMeta() {
  const meta = state.data.meta;
  const instance = meta.instance || {};
  $("meta").textContent = [
    meta.network,
    `${meta.customers} customers over the real Bengaluru network`,
    `budget ${meta.budget.toLocaleString()} evaluations · seed ${meta.seed} · SUMO seed ${meta.sumo_seed}`,
    `${(meta.sumo_steps / 3600).toFixed(1)} h simulated per method`,
  ].join("\n");
}

/* --------------------------------------------------------------------- cards */

function renderCards() {
  const host = $("cards");
  host.innerHTML = "";
  const ranked = rankedRuns();
  const best = ranked.length ? ranked[0].objective : Infinity;
  for (const run of state.data.runs) {
    const card = document.createElement("button");
    card.className = "card";
    card.type = "button";
    card.dataset.key = run.key;
    const gap = run.objective < best + 1e-9;
    const sim = run.simulation && !run.simulation.error ? run.simulation : null;
    card.innerHTML = `
      <div class="card-top">
        <span class="card-name"></span>
        <span class="fam ${run.family}">${FAMILY_LABEL[run.family] || run.family}</span>
      </div>
      <div class="card-obj"></div>
      <div class="card-sub">
        <span>${run.vehicles} veh · ${(modelledHoursPerVehicle(run) * 60).toFixed(0)} min/veh modelled</span>
        <span class="rank ${gap ? "r1" : ""}">${gap ? "best" : `+${pct(run.objective, best)}`}</span>
      </div>
      <div class="card-sub">
        <span>${sim ? `SUMO <b>${(sim.mean_travel_time_s / 60).toFixed(0)} min/veh</b>` : "no SUMO run"}</span>
        <span>${run.wall_time.toFixed(2)} s</span>
      </div>`;
    card.querySelector(".card-name").textContent = run.name;
    card.querySelector(".card-obj").textContent = fmt(run.objective);
    if (run.error) card.querySelector(".card-obj").textContent = "failed";
    card.setAttribute("aria-pressed", "false");
    card.setAttribute("aria-label", `show ${run.name} on the map`);
    card.addEventListener("click", () => select(run.key));
    host.appendChild(card);
  }
}

function rankedRuns() {
  return state.data.runs
    .filter((r) => !r.error && isFinite(r.objective))
    .sort((a, b) => a.objective - b.objective);
}

function pct(value, best) {
  if (!isFinite(best) || best === 0) return "n/a";
  return `${(((value - best) / best) * 100).toFixed(1)}%`;
}

/* ----------------------------------------------------------------- selection */

function select(key) {
  state.selected = key;
  for (const card of document.querySelectorAll(".card")) {
    const on = card.dataset.key === key;
    card.classList.toggle("sel", on);
    card.setAttribute("aria-pressed", String(on));
  }
  for (const row of document.querySelectorAll("#table tbody tr")) {
    row.classList.toggle("sel", row.dataset.key === key);
  }
  const run = currentRun();
  if (!run) return;
  $("map-title").textContent = `${run.name}, ${run.vehicles} vehicle routes`;
  const driven = simOf(run, "mean_distance_m", true);
  $("map-sub").textContent =
    `${state.data.meta.customers} customers, ${run.polylines.reduce((n, p) => n + p.length, 0)} junctions driven, ` +
    `${(run.eval_distance / 1000).toFixed(1)} km priced, ` +
    // A missing measurement is not zero kilometres driven.  Dividing the null by
    // 1000 would report "0.0 km driven" for a run SUMO never completed.
    (driven === null ? "no SUMO measurement" : `${(driven / 1000).toFixed(1)} km driven`);
  renderMeasured(run);
  drawLegend(run);
  $("gui").disabled = !run.routes_file;
  state.frame = 0;
  state.lastTick = 0;
  renderLegs(run);
  draw();
}

function currentRun() {
  return state.data.runs.find((r) => r.key === state.selected) || null;
}

/* ----------------------------------------------------------------- projection */

function buildProjection() {
  const [minX, minY, maxX, maxY] = state.data.meta.bounds;
  const canvas = $("map");
  const rect = canvas.getBoundingClientRect();
  const width = rect.width || 1200;
  const height = rect.height || 520;
  const canvasRatio = width / height;
  const dataRatio = (maxX - minX) / (maxY - minY);
  let scale;
  let offsetX;
  let offsetY;
  if (dataRatio > canvasRatio) {
    scale = width / (maxX - minX);
    offsetX = 0;
    offsetY = (height - (maxY - minY) * scale) / 2;
  } else {
    scale = height / (maxY - minY);
    offsetY = 0;
    offsetX = (width - (maxX - minX) * scale) / 2;
  }
  state.projection = {
    scale,
    project: (x, y) => [x * scale + offsetX, height - (y * scale + offsetY)],
  };
  // One Path2D per rank, painted trunk first, so arterial roads draw over the
  // service streets they intersect. Path2D.addPath would keep insertion order
  // instead of letting the ranks be stacked deliberately.
  state.basemaps = [0, 1, 2, 3].map((rank) => {
    const path = new Path2D();
    for (const row of state.data.basemap) {
      if (row[4] !== rank) continue;
      const [x1, y1] = state.projection.project(row[0], row[1]);
      const [x2, y2] = state.projection.project(row[2], row[3]);
      path.moveTo(x1, y1);
      path.lineTo(x2, y2);
    }
    return path;
  });
  state.basemapColours = ["#2f3b4e", "#28323f", "#212a36", "#1b232e"];
  state.basemapWidths = [1.8, 1.4, 1.0, 0.7];
}

/* Track keys are vehicle identities such as "cargoTruck_3", not array indices.
 * Taking the trailing number is what keeps a vehicle the same colour in the
 * animation, the legend and the per-vehicle bars; indexing the palette with the
 * whole key yields NaN and an undefined colour. */
function vehicleColour(vid) {
  const n = Number(vid.match(/(\d+)\s*$/)?.[1] ?? 0);
  return VEHICLE_COLOURS[n % VEHICLE_COLOURS.length];
}

function drawLegend(run) {
  const items = [
    ...run.polylines.map((_, i) => ({ colour: vehicleColour(`v${i}`), label: `vehicle ${i + 1}` })),
    { colour: "#ffd166", label: "depot" },
    { colour: "#e6edf6", label: "customer" },
  ];
  $("legend").innerHTML = items
    .map((i) => `<span><i class="swatch" style="background:${i.colour}"></i>${i.label}</span>`)
    .join("");
}

/* ----------------------------------------------------------------------- map */

function draw() {
  const canvas = $("map");
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (canvas.width !== Math.round(rect.width * ratio)) {
    canvas.width = Math.round(rect.width * ratio);
    canvas.height = Math.round(rect.height * ratio);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!state.projection) return;

  if ($("show-base").checked) {
    state.basemaps.forEach((path, rank) => {
      ctx.strokeStyle = state.basemapColours[rank];
      ctx.lineWidth = state.basemapWidths[rank];
      ctx.stroke(path);
    });
  }

  const run = currentRun();
  if (!run) return;
  const { project } = state.projection;

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  run.polylines.forEach((points, index) => {
    if (points.length < 2) return;
    const colour = vehicleColour(`v${index}`);
    ctx.strokeStyle = colour;
    ctx.globalAlpha = 0.32;
    ctx.lineWidth = 6.5;
    tracePath(ctx, points, project);
    ctx.stroke();
    ctx.globalAlpha = 0.95;
    ctx.lineWidth = 2.1;
    tracePath(ctx, points, project);
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  const [depotX, depotY] = state.data.meta.depot;
  const points = state.data.meta.points;
  if ($("show-points").checked) {
    ctx.fillStyle = "rgba(230,237,246,.62)";
    for (const [x, y] of points.slice(1)) {
      const [px, py] = project(x, y);
      ctx.beginPath();
      ctx.arc(px, py, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#ffd166";
    ctx.beginPath();
    ctx.arc(...project(depotX, depotY), 6.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#0e1117";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  if ($("show-anim").checked) drawFleet(ctx, run);
}

function tracePath(ctx, points, project) {
  ctx.beginPath();
  points.forEach(([x, y], i) => {
    const [px, py] = project(x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
}

function drawFleet(ctx, run) {
  const track = run.simulation && run.simulation.trace;
  if (!track) return;
  const { project } = state.projection;
  const now = state.frame;
  for (const [vid, points] of Object.entries(track.tracks)) {
    const here = sampleTrack(points, now);
    if (!here) continue;
    const [x, y, speed] = here;
    const [px, py] = project(x, y);
    const colour = vehicleColour(vid);
    ctx.beginPath();
    ctx.arc(px, py, 7.5, 0, Math.PI * 2);
    ctx.fillStyle = hexAlpha(colour, 0.22);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, 3.6, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "#0e1117";
    ctx.stroke();
    if (speed > 0.4) {
      ctx.beginPath();
      ctx.arc(px, py, 10.5, 0, Math.PI * 2);
      ctx.strokeStyle = hexAlpha(colour, 0.45 * Math.min(1, speed / 12));
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

function hexAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

/* ------------------------------------------------------- where vehicles are going */

/* A trace key is "<profile>_<index>" and a polyline is indexed by position in the
 * route list.  Those are the same number, but only once the trailing digits are
 * read out of the key; comparing the whole key as an index silently matches
 * nothing, which is how a "next stop" panel ends up permanently empty. */
function vehicleIndex(vid) {
  const n = Number(vid.match(/(\d+)\s*$/)?.[1] ?? NaN);
  return Number.isFinite(n) ? n : -1;
}

function traceKeyFor(run, index) {
  const tracks = run.simulation && run.simulation.trace && run.simulation.trace.tracks;
  if (!tracks) return null;
  return Object.keys(tracks).find((k) => vehicleIndex(k) === index) ?? null;
}

function nearestOnPath(path, x, y) {
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < path.length; i += 1) {
    const dx = path[i][0] - x;
    const dy = path[i][1] - y;
    const d = dx * dx + dy * dy;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/* Where one vehicle is on its route at simulation time `now`.
 *
 * The stop sequence is exact, but working out which stop a vehicle is *on* means
 * mapping its position back onto the drawn path, and the two disagree on
 * purpose: the polyline is turn-exact while the vehicle sits inside a lane or a
 * junction.  Snapping to the nearest path point is sound here because
 * consecutive stops are whole legs apart, so a junction-sized error cannot move a
 * vehicle onto a different leg — it only makes the reading flip where the answer
 * is genuinely ambiguous, which is on the stop itself. */
function legProgress(run, index, now) {
  const marks = (run.waypoints || [])[index];
  const path = (run.polylines || [])[index];
  if (!marks || marks.length < 2 || !path || !path.length) return null;
  const key = traceKeyFor(run, index);
  if (!key) return null;

  const track = run.simulation.trace.tracks[key];
  const here = sampleTrack(track, now);
  let at;
  let status;
  if (here) {
    at = nearestOnPath(path, here[0], here[1]);
    status = here[2] > 0.4 ? "en route" : "stopped";
  } else if (track.length && now < track[0][0]) {
    at = 0;
    status = "at depot";
  } else {
    at = path.length - 1;
    const done = (run.simulation.per_vehicle || {})[key] || {};
    status = done.finished ? "route complete" : "simulation ended";
  }

  let leg = 0;
  while (leg + 2 < marks.length && marks[leg + 1].at <= at) leg += 1;
  const from = marks[leg];
  const to = marks[leg + 1];
  const span = to.at - from.at;
  const done = span > 0 ? Math.min(1, Math.max(0, (at - from.at) / span)) : 1;
  return {
    key,
    index,
    from,
    to,
    done,
    reached: marks.filter((m) => m.kind === "customer" && m.at <= at).length,
    stops: marks.filter((m) => m.kind === "customer").length,
    speed: here ? here[2] : 0,
    status,
  };
}

function legCard(leg) {
  const node = (m) => `<span class="leg-node">${escapeHtml(m.label)}</span>`;
  return (
    `<div class="leg" style="--v:${vehicleColour(leg.key)}">` +
    `<div class="leg-head">` +
    `<span class="leg-name">${escapeHtml(leg.key)}</span>` +
    `<span class="leg-count">${escapeHtml(leg.status)} · ${leg.reached}/${leg.stops} stops</span>` +
    `</div>` +
    `<div class="leg-route">` +
    `<span class="leg-side"><span class="leg-cap">last stop</span>${node(leg.from)}</span>` +
    `<span class="leg-arrow">&rarr;</span>` +
    `<span class="leg-side"><span class="leg-cap">next stop</span>${node(leg.to)}</span>` +
    `<span class="leg-speed">${(leg.speed * 3.6).toFixed(0)} km/h</span>` +
    `</div>` +
    `<div class="leg-bar"><i style="width:${(leg.done * 100).toFixed(1)}%"></i></div>` +
    `</div>`
  );
}

/* One innerHTML write per frame for a handful of small cards.  The panel is the
 * only thing on the page that changes frame to frame, and rewriting six short
 * strings is cheaper than the rest of the frame — a node-reuse optimisation here
 * would only be worth its own testing. */
function renderLegs(run) {
  const cards = [];
  (run.polylines || []).forEach((_, index) => {
    const leg = legProgress(run, index, state.frame);
    if (leg) cards.push(legCard(leg));
  });
  state.legs = cards.length;
  $("fleet-legs").innerHTML = cards.length
    ? cards.join("")
    : `<p class="leg-none">This method's routes were not simulated, so there is no vehicle ` +
      `position to read a stop sequence from. Pick a method with a SUMO run to follow ` +
      `its vehicles.</p>`;
}

function sampleTrack(points, now) {
  if (!points.length || now < points[0][0] || now > points[points.length - 1][0]) return null;
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] <= now) lo = mid;
    else hi = mid;
  }
  const a = points[lo];
  const b = points[hi];
  const span = b[0] - a[0] || 1;
  const f = (now - a[0]) / span;
  return [a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f];
}

/* ------------------------------------------------------------------ playback */

function tick(now) {
  const run = currentRun();
  const horizon = run && run.simulation && run.simulation.trace ? run.simulation.trace.steps : 0;
  if (horizon) {
    if (state.playing) {
      if (!state.lastTick) state.lastTick = now;
      const elapsed = (now - state.lastTick) / 1000;
      state.lastTick = now;
      state.frame = (state.frame + elapsed * 90) % horizon;
      $("scrub").value = String(state.frame / horizon);
    } else {
      state.lastTick = now;
    }
    $("scrub-label").textContent = `t = ${Math.round(state.frame / 60)} min`;
    draw();
    renderLegs(run);
  }
  requestAnimationFrame(tick);
}

$("play").addEventListener("click", () => {
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "pause" : "play";
});

$("scrub").addEventListener("input", (event) => {
  const run = currentRun();
  const horizon = run && run.simulation && run.simulation.trace ? run.simulation.trace.steps : 0;
  state.frame = Number(event.target.value) * horizon;
  state.playing = false;
  $("play").textContent = "play";
  draw();
  renderLegs(run);
});

for (const id of ["show-base", "show-points", "show-anim"]) {
  $(id).addEventListener("change", draw);
}

window.addEventListener("resize", () => {
  if (!state.data) return;
  buildProjection();
  draw();
  drawConvergence();
});

$("gui").addEventListener("click", async () => {
  const run = currentRun();
  if (!run) return;
  const button = $("gui");
  button.disabled = true;
  try {
    const response = await fetch("/api/sumo-gui", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ net_file: netFile(), routes_file: run.routes_file }),
    });
    const payload = await response.json();
    if (payload.error) window.alert(payload.error);
  } finally {
    button.disabled = false;
  }
});

function netFile() {
  return state.data.meta.net_file || `data/networks/net/${state.data.meta.network}.net.xml`;
}

/* --------------------------------------------------------------- measurement */

function renderMeasured(run) {
  const sim = run.simulation;
  const host = $("measured");
  if (!sim) {
    host.innerHTML = '<p>This method did not produce a simulable solution.</p>';
    return;
  }
  if (sim.error) {
    host.innerHTML = `<p>SUMO could not run this solution:</p><pre class="error">${escapeHtml(sim.error)}</pre>`;
    return;
  }
  const fleet = Object.entries(sim.per_vehicle || {})
    .map(([vid, v]) => ({ vid, value: v.travel_time }))
    .filter((row) => isFinite(row.value))
    .sort((a, b) => b.value - a.value);
  // Scaling to half the fleet mean times two put every bar in an arbitrary place:
  // the length depended on the mean rather than on anything in the set, so two
  // vehicles with the same journey time could draw different bars and a change
  // of method could rescale every bar without changing any journey.  The full
  // bar is now the slowest vehicle actually in the set, with a tick on the mean
  // so a single slow vehicle is visible against the rest.
  const slowest = fleet.length ? fleet[0].value : 0;
  const meanAt = slowest ? (sim.mean_travel_time_s / slowest) * 100 : 0;
  host.innerHTML = `
    <div class="kpis">
      ${kpi("mean journey", `${(sim.mean_travel_time_s / 60).toFixed(1)} min`, `${sim.finished_vehicles} of ${sim.vehicles} vehicles finished`)}
      ${kpi("total delay", `${(sim.total_time_loss_s / 60).toFixed(0)} min`, "against SUMO's free-flow reference")}
      ${kpi("distance driven", `${(sim.mean_distance_m / 1000).toFixed(1)} km`, "mean per vehicle, odometer")}
      ${kpi("CO₂", `${(sim.total_co2_g / 1000).toFixed(1)} kg`, "fleet total, SUMO emission model")}
      ${kpi("completion", `${(sim.completion_rate * 100).toFixed(0)}%`, `${sim.arrivals} arrived of ${sim.departures} departed`)}
      ${kpi("network speed", `${sim.mean_network_speed_kmh.toFixed(1)} km/h`, "SUMO's mean over every edge, not just ours")}
    </div>
    <div class="bar-group-title">
      <span>per vehicle, worst first</span>
      <span class="bar-scale">${
        fleet.length
          ? `full bar = ${(slowest / 60).toFixed(0)} min, the slowest · tick = fleet mean`
          : "no per-vehicle records"
      }</span>
    </div>
    ${fleet
      .map(({ vid, value }) => {
        const colour = vehicleColour(vid);
        return `<div class="bar-row">
          <span class="bar-name">${escapeHtml(vid)}</span>
          <span class="bar-track">
            <span class="bar-fill" style="width:${(value / slowest) * 100}%;background:${colour}"></span>
            ${meanAt > 0 && meanAt < 100 ? `<span class="bar-tick" style="left:${meanAt}%"></span>` : ""}
          </span>
          <span class="bar-val">${(value / 60).toFixed(0)} min</span>
        </div>`;
      })
      .join("")}`;
}

/* Every quantity on this page is lower-is-better, so a longer bar is a worse
 * result.  Each group is therefore sorted best-first and scaled to its own
 * worst value, and that worst value is printed: a bar whose length is relative
 * to an unstated reference cannot be read, and cannot be compared with a bar in
 * the group above it. */
const BAR_GROUPS = [
  { title: "objective, what the optimiser minimised", value: (r) => r.objective, fmt: (v) => fmt(v) },
  {
    title: "distance per vehicle, SUMO's odometer",
    value: (r) => simOf(r, "mean_distance_m", true),
    fmt: (v) => `${(v / 1000).toFixed(1)} km`,
  },
  {
    title: "delay per vehicle, SUMO against free flow",
    value: (r) => perFinished("total_time_loss_s", 1)(r),
    fmt: (v) => `${(v / 60).toFixed(1)} min`,
  },
  {
    title: "CO₂ per vehicle, SUMO emission model",
    value: (r) => perFinished("total_co2_g", 1)(r),
    fmt: (v) => `${(v / 1000).toFixed(2)} kg`,
  },
];

function kpi(label, value, note) {
  return `<div class="kpi">
    <div class="kpi-label">${label}</div>
    <div class="kpi-value">${value}</div>
    <div class="kpi-note">${note}</div>
  </div>`;
}

/* --------------------------------------------------------------- head-to-head */

function renderHeadToHead() {
  const runs = state.data.runs.filter((r) => !r.error);
  $("bars").innerHTML = BAR_GROUPS.map((group) => {
    // A method whose routes SUMO could not drive has no value here.  Leaving it
    // out of both the sort and the scale is the point: scoring the absence as
    // zero would draw an empty bar that reads as the best result on the page.
    const rows = runs
      .map((run) => ({ run, value: group.value(run) }))
      .filter((row) => row.value !== null && isFinite(row.value))
      .sort((a, b) => a.value - b.value);
    const dropped = runs.length - rows.length;
    if (!rows.length) {
      return `<div class="bar-group">${barGroupTitle(group.title, null, dropped)}` +
        `<p class="bar-empty">no method produced a measurement for this</p></div>`;
    }
    const worst = rows[rows.length - 1].value;
    return `<div class="bar-group">${barGroupTitle(group.title, group.fmt(worst), dropped)}` +
      rows
        .map(({ run, value }) => {
          const quantum = run.family !== "classical";
          return `<div class="bar-row">
            <span class="bar-name">${escapeHtml(run.name)}</span>
            <span class="bar-track"><span class="bar-fill" style="width:${(value / worst) * 100}%;background:${quantum ? "#4cc9f0" : "#f4a261"};opacity:${quantum ? ".85" : ".7"}"></span></span>
            <span class="bar-val">${group.fmt(value)}</span>
          </div>`;
        })
        .join("") +
      "</div>";
  }).join("");
  $("verdict").textContent = verdictText();
}

function barGroupTitle(title, worstLabel, dropped) {
  const scale = worstLabel
    ? `best first · full bar = ${worstLabel}, the worst here`
    : "nothing to compare";
  const missing = dropped ? ` · ${dropped} not measured` : "";
  return `<div class="bar-group-title"><span>${title}</span><span class="bar-scale">${scale}${missing}</span></div>`;
}

function simOf(run, field, strict) {
  const sim = run.simulation;
  if (!sim || sim.error || !(field in sim)) return strict ? null : 0;
  return sim[field];
}

function verdictText() {
  const quantum = state.data.runs.filter((r) => r.family !== "classical" && !r.error);
  const classical = state.data.runs.filter((r) => r.family === "classical" && !r.error);
  if (!quantum.length || !classical.length) return "Not enough successful runs to compare.";
  const mean = (rows, f) => rows.reduce((a, r) => a + f(r), 0) / rows.length;
  const parts = [
    ["objective", "objective", 1],
    ["distance per vehicle", "mean_distance_m", 1000, "km"],
    ["time loss", "total_time_loss_s", 60, "min"],
    ["CO₂", "total_co2_g", 1000, "kg"],
  ]
    .map(([label, field, divisor, unit]) => {
      const a = mean(quantum, (r) => (field === "objective" ? r.objective : simOf(r, field)));
      const b = mean(classical, (r) => (field === "objective" ? r.objective : simOf(r, field)));
      if (!a || !b) return null;
      const delta = ((b - a) / b) * 100;
      return `${label} ${delta >= 0 ? "−" : "+"}${Math.abs(delta).toFixed(1)}%`;
    })
    .filter(Boolean)
    .join(", ");
  return `Quantum-inspired methods averaged ${parts} against the classical baselines, on the same instance, budget and simulated network.`;
}

/* ---------------------------------------------------------------- convergence */

function buildConvergence() {
  const runs = state.data.runs.filter((r) => r.trace_costs && r.trace_costs.length > 1);
  if (!runs.length) return null;
  const all = runs.flatMap((r) => r.trace_costs).filter(isFinite);
  const maxEvals = Math.max(...runs.map((r) => r.trace_evals[r.trace_evals.length - 1]));
  return {
    runs,
    maxEvals,
    minCost: Math.min(...all),
    maxCost: Math.max(...all),
  };
}

function renderConvergence() {
  const model = buildConvergence();
  if (!model) {
    $("convergence").closest(".panel").hidden = true;
    return;
  }
  state.convergence = model;
  drawConvergence();
}

function drawConvergence() {
  const model = state.convergence;
  const canvas = $("convergence");
  if (!model) return;
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);

  const pad = { left: 62, right: 18, top: 14, bottom: 32 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const span = Math.max(model.maxCost - model.minCost, 1e-9);
  const lo = model.minCost - span * 0.06;
  const hi = model.maxCost + span * 0.06;
  const x = (v) => pad.left + (v / model.maxEvals) * plotW;
  const y = (v) => pad.top + plotH - ((v - lo) / (hi - lo)) * plotH;

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = "#93a1b5";
  ctx.strokeStyle = "#26303f";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const value = lo + ((hi - lo) * i) / 4;
    const py = y(value);
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(width - pad.right, py);
    ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(shortNum(value), pad.left - 8, py + 3.5);
  }
  for (let i = 0; i <= 4; i += 1) {
    const value = (model.maxEvals * i) / 4;
    ctx.textAlign = "center";
    ctx.fillText(String(Math.round(value)), x(value), height - 12);
  }
  ctx.textAlign = "center";
  ctx.fillText("evaluations", pad.left + plotW / 2, height - 0.5);

  for (const run of model.runs) {
    const quantum = run.family !== "classical";
    ctx.strokeStyle = quantum ? "#4cc9f0" : "#f4a261";
    ctx.globalAlpha = run.key === state.selected ? 1 : 0.5;
    ctx.lineWidth = run.key === state.selected ? 2.6 : 1.4;
    ctx.setLineDash(quantum ? [] : [5, 4]);
    ctx.beginPath();
    run.trace_costs.forEach((cost, i) => {
      const px = x(run.trace_evals[i]);
      const py = y(cost);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  let cursorY = pad.top + 4;
  ctx.textAlign = "left";
  for (const run of model.runs.slice(0, 12)) {
    ctx.fillStyle = run.family === "classical" ? "#f4a261" : "#4cc9f0";
    ctx.fillRect(width - pad.right - 116, cursorY, 12, 2.5);
    ctx.fillStyle = "#93a1b5";
    ctx.fillText(run.name, width - pad.right - 99, cursorY + 4);
    cursorY += 14;
  }
}

/* --------------------------------------------------------------------- table */

/* A column reads null when its source is missing — a method whose routes SUMO
 * could not drive has no distance, and scoring that absence as zero would hand
 * it the best cell in the table. */

/* The modelled columns are a fleet total and the SUMO columns are a mean per
 * vehicle, so both are put on a per-vehicle basis before being shown next to
 * each other.  Comparing a fleet total against a per-vehicle mean invites a
 * reading wrong by the fleet size, and inviting that comparison is the whole
 * purpose of the two column groups. */
function modelledHoursPerVehicle(run) {
  return run.vehicles > 0 ? run.eval_time / run.vehicles / 3600 : null;
}

function modelledKmPerVehicle(run) {
  return run.vehicles > 0 ? run.eval_distance / run.vehicles / 1000 : null;
}

function measuredHoursPerVehicle(run) {
  const seconds = simOf(run, "mean_travel_time_s", true);
  return seconds === null ? null : seconds / 3600;
}

function perFinished(sumField, scale) {
  return (run) => {
    const sum = simOf(run, sumField, true);
    const done = simOf(run, "finished_vehicles", true);
    if (sum === null || !done) return null;
    return sum / done / scale;
  };
}

/* cellText divides by `scale`, so a column that is already a fraction has to
 * scale itself here rather than through `scale`, or a completion rate of 1.0
 * renders as 0. */
function completionPct(run) {
  const rate = simOf(run, "completion_rate", true);
  return rate === null ? null : rate * 100;
}

/* How far the surrogate's travel time sat from what SUMO measured.  This is the
 * number that says whether the cost model can be trusted, so it is stated rather
 * than left for the reader to derive from two columns. */
function modelErrorPct(run) {
  const modelled = modelledHoursPerVehicle(run);
  const measured = measuredHoursPerVehicle(run);
  if (modelled === null || measured === null || modelled === 0) return null;
  return ((measured - modelled) / modelled) * 100;
}

const COLUMNS = [
  { label: "method", value: (r) => r.name, text: true },
  { label: "family", value: (r) => FAMILY_LABEL[r.family] || r.family, text: true },
  { label: "objective", value: (r) => r.objective, best: "min", group: "modelled", sep: true },
  { label: "vehicles", value: (r) => r.vehicles, best: "min", group: "modelled" },
  { label: "unserved", value: (r) => r.unserved, best: "min", group: "modelled" },
  { label: "h/veh", value: modelledHoursPerVehicle, digits: 2, best: "min", group: "modelled" },
  { label: "km/veh", value: modelledKmPerVehicle, digits: 1, best: "min", group: "modelled" },
  { label: "evals", value: (r) => r.n_evals, group: "modelled" },
  { label: "s", value: (r) => r.wall_time, digits: 2, best: "min", group: "modelled" },
  { label: "h/veh", value: measuredHoursPerVehicle, digits: 2, best: "min", group: "SUMO", sep: true },
  { label: "km/veh", value: (r) => simOf(r, "mean_distance_m", true), scale: 1000, digits: 1, best: "min", group: "SUMO" },
  { label: "delay min/veh", value: perFinished("total_time_loss_s", 60), digits: 1, best: "min", group: "SUMO" },
  { label: "CO₂ kg/veh", value: perFinished("total_co2_g", 1000), digits: 2, best: "min", group: "SUMO" },
  { label: "done %", value: completionPct, digits: 0, best: "max", group: "SUMO" },
  { label: "sim s", value: (r) => simOf(r, "wall_time_s", true), digits: 0, group: "SUMO" },
  { label: "model err %", value: modelErrorPct, digits: 0, group: "surrogate", sep: true },
];

function cellText(column, value) {
  if (value === null || !isFinite(value)) return "n/a";
  const scaled = value / (column.scale || 1);
  return column.digits === undefined ? scaled.toLocaleString() : scaled.toFixed(column.digits);
}

function renderTable() {
  const table = $("table");
  // A caption row naming each group, so "modelled" and "SUMO" are visibly two
  // different sources of truth rather than two more columns of numbers.
  const GROUP_TITLES = { modelled: "optimiser", SUMO: "measured in SUMO", surrogate: "surrogate fidelity" };
  const groupRow = COLUMNS.map((c, i) => {
    if (!c.group) return "";
    if (c.group === COLUMNS[i - 1]?.group) return "";
    return `<th scope="colgroup" class="g start" colspan="${groupSpan(COLUMNS, i)}">${GROUP_TITLES[c.group] || c.group}</th>`;
  }).join("");
  const head = COLUMNS.map((c) => `<th scope="col" class="${c.sep ? "sep" : ""} ${c.group ? "ing" : ""}">${c.label}</th>`).join("");
  const rows = state.data.runs
    .map((run) => {
      const cells = COLUMNS.map((c) => {
        const raw = c.value(run);
        const shown = c.text ? (raw === null ? "n/a" : escapeHtml(raw)) : cellText(c, raw);
        const good = c.best && isExtreme(c, run) ? "best" : "";
        return `<td class="${c.sep ? "sep" : ""} ${good}">${shown}</td>`;
      }).join("");
      return `<tr data-key="${run.key}" tabindex="0" aria-label="show ${escapeHtml(run.name)} on the map">${cells}</tr>`;
    })
    .join("");
  // The caption is what a screen reader announces for the table, so it has to
  // carry the two facts the column headers alone do not: that the groups are
  // different sources of truth, and that every column is per vehicle.
  const caption =
    "<caption class=\"sr-only\">Every method on the same instance, evaluation budget and " +
    "simulated network. The optimiser columns are what the search minimised; the SUMO " +
    "columns are what the same routes measured when driven. Time and distance are per " +
    "vehicle in both groups, because the objective is a fleet total and the measurements " +
    "are per-vehicle means. Lower is better except completion.</caption>";
  table.innerHTML = `${caption}<thead><tr class="groups">${groupRow}</tr><tr>${head}</tr></thead><tbody>${rows}</tbody>`;
  // The rows are the second way to pick a method, and a row is not a button, so
  // it needs a tab stop and a key handler or the whole comparison table is
  // mouse-only.
  for (const row of table.querySelectorAll("tbody tr")) {
    row.addEventListener("click", () => select(row.dataset.key));
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      select(row.dataset.key);
    });
  }
  select(state.selected);
}

function groupSpan(columns, start) {
  const group = columns[start].group;
  let span = 0;
  for (let i = start; i < columns.length && columns[i].group === group; i += 1) span += 1;
  return span;
}

function isExtreme(column, run) {
  const values = state.data.runs.map((r) => column.value(r)).filter((v) => v !== null && isFinite(v));
  if (!values.length) return false;
  const target = column.best === "min" ? Math.min(...values) : Math.max(...values);
  return Math.abs(column.value(run) - target) < 1e-9;
}

/* -------------------------------------------------------------------- format */

function fmt(value) {
  if (!isFinite(value)) return "n/a";
  if (value >= 100000) return `${(value / 1000).toFixed(1)}k`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function shortNum(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toFixed(0);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

boot();
