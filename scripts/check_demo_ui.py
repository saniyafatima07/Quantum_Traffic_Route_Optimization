"""Exercise the demo page's JavaScript against a real payload, without a browser.

``app.js`` is ordinary DOM code, so the honest way to check it is to run it.  A
minimal stub stands in for the browser: enough of ``document``, ``canvas`` and
``fetch`` for every render path to execute against a comparison the server
actually produced.  Anything that throws here would throw in the page, and this
runs in a second where a browser session takes a minute and needs a display.

Run with:  uv run python scripts/check_demo_ui.py [payload.json]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "qitransit" / "demo" / "static"

HARNESS = r"""
const fs = require("fs");
const vm = require("vm");

const payloadPath = process.argv[2];
const payload = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
const html = fs.readFileSync(process.argv[3], "utf8");
const source = fs.readFileSync(process.argv[4], "utf8");

const elementIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
const classes = new Set(
  [...html.matchAll(/class="([^"]+)"/g)].flatMap((m) => m[1].split(/\s+/))
);

class Stub {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.classList = { toggle() {}, add() {}, remove() {} };
    this._classes = new Set();
    this.innerHTML = "";
    this.textContent = "";
    this.hidden = false;
    this.value = "0";
    this.disabled = false;
  }
  appendChild(child) { this.children.push(child); return child; }
  /* Containers assembled by appending hold no innerHTML, because innerHTML is
   * only ever what a string assignment left behind.  Without this the dump is
   * empty for exactly the markup JavaScript builds node by node, which is where
   * the method cards live, so every check that reads the dump is blind to them. */
  markup() {
    const kids = this.children.length
      ? this.children.map((child) => (typeof child.markup === "function" ? child.markup() : "")).join("")
      : this.innerHTML || "";
    if (!kids) return "";
    const named = typeof this.className === "string" && this.className ? ` class="${this.className}"` : "";
    return `<${this.tagName}${named}>${kids}</${this.tagName}>`;
  }
  addEventListener() {}
  removeEventListener() {}
  setAttribute(k, v) { this[k] = v; }
  // Attributes are stored flat, so a read has to look in the same place a write
  // put them or every accessibility assertion against this stub is vacuous.
  getAttribute(k) { return Object.prototype.hasOwnProperty.call(this, k) ? this[k] : null; }
  getBoundingClientRect() { return { width: 1200, height: 520 }; }
  getContext() { return context2d(); }
  closest() { return new Stub(); }
  querySelector() { return new Stub(); }
  querySelectorAll() { return []; }
}

const elements = new Map();
for (const id of elementIds) elements.set(id, new Stub());

function context2d() {
  const noop = () => {};
  return {
    setTransform: noop, clearRect: noop, beginPath: noop, moveTo: noop,
    lineTo: noop, stroke: noop, fill: noop, arc: noop, rect: noop,
    fillRect: noop, strokeRect: noop, clip: noop, translate: noop, scale: noop,
    fillText: noop, strokeText: noop, setLineDash: noop, save: noop, restore: noop,
    measureText: () => ({ width: 10 }),
    canvas: { width: 1200, height: 520 },
  };
}

class Path2DStub {
  constructor() { this.ops = 0; }
  moveTo() { this.ops += 1; }
  lineTo() { this.ops += 1; }
  addPath(other) { this.ops += other.ops; }
}

const document_ = {
  getElementById(id) {
    if (!elements.has(id)) throw new Error("missing element id: " + id);
    return elements.get(id);
  },
  querySelectorAll() { return []; },
  createElement(tag) { return new Stub(tag); },
};

let frames = 0;
const sandbox = {
  console,
  document: document_,
  Path2D: Path2DStub,
  Math,
  JSON,
  Object,
  Array,
  String,
  Number,
  isFinite,
  parseInt,
  parseFloat,
  setTimeout: () => 0,
  devicePixelRatio: 2,
  window: { devicePixelRatio: 2, addEventListener() {} },
  requestAnimationFrame(fn) { if (frames++ < 3) fn(16.7); return frames; },
  fetch(url) {
    if (url === "/api/state") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ ready: true }) });
    }
    if (url === "/api/data") {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(payload) });
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
  },
};
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
// `const` and `function` declarations in a vm script are lexical, not properties
// of the sandbox object, so reach the page's internals through an epilogue.
vm.runInContext(
    source + "\nglobalThis.__state = state;\nglobalThis.__select = select;\n"
        + "globalThis.__draw = draw;\nglobalThis.__vehicleColour = vehicleColour;\n"
        + "globalThis.__renderHeadToHead = renderHeadToHead;\n"
        + "globalThis.__renderLegs = renderLegs;\n"
        + "globalThis.__legProgress = legProgress;\n",
    sandbox,
    { filename: "app.js" },
);

setTimeout(() => {
  const state = sandbox.__state;
  const checks = {
    "payload loaded": Boolean(state && state.data),
    "a method is selected": Boolean(state.selected),
    "projection built": Boolean(state.projection && state.projection.scale > 0),
    "basemap pathed": Boolean(state.basemaps && state.basemaps.reduce((n, p) => n + p.ops, 0) > 100),
    "convergence modelled": Boolean(state.convergence),
    "table rendered": elements.get("table").innerHTML.includes("</tr>"),
    "group captions": elements.get("table").innerHTML.includes("measured in SUMO"),
    "per-vehicle model column": elements.get("table").innerHTML.includes("h/veh"),
    "surrogate error column": elements.get("table").innerHTML.includes("model err %"),
    "table has a caption": elements.get("table").innerHTML.includes("<caption"),
    "every header has a scope": (() => {
      const html = elements.get("table").innerHTML;
      const total = (html.match(/<th[ >]/g) || []).length;
      const scoped = (html.match(/<th scope="(col|colgroup)"/g) || []).length;
      return total > 0 && total === scoped;
    })(),
    "rows are keyboard reachable": (() => {
      const html = elements.get("table").innerHTML;
      return (html.match(/<tr data-key="[^"]*" tabindex="0"/g) || []).length === payload.runs.length;
    })(),
    "cards expose their state": (() => {
      const host = elements.get("cards");
      return [...host.children].every(
        (c) => c.getAttribute("aria-pressed") !== null && c.getAttribute("aria-label")
      );
    })(),
    "cards rendered": elements.get("cards").children.length === payload.runs.length,
    "map title set": elements.get("map-title").textContent.length > 0,
    "kpis rendered": elements.get("measured").innerHTML.includes("kpi-value"),
    "bars rendered": elements.get("bars").innerHTML.includes("bar-fill"),
    "legend rendered": elements.get("legend").innerHTML.includes("swatch"),
  };

  // A bar is only readable if its length is a stated fraction of a stated
  // quantity.  These check the three properties that make a bar chart a bar
  // chart: the worst entry fills the track, the group is ordered so the ranking
  // can be read off it, and nothing is drawn as an empty bar that would read as
  // the best result.
  const readBars = () => {
    const host = elements.get("bars");
    const groups = [];
    let current = null;
    for (const node of host.innerHTML.split(/<div class="bar-group">/)) {
      const rows = [...node.matchAll(/<span class="bar-val">([^<]*)<\/span>/g)].map((m) => m[1]);
      if (rows.length) groups.push({ rows, html: node });
    }
    return groups;
  };

  const barWidths = (group) =>
    [...group.html.matchAll(/class="bar-fill" style="width:([0-9.e-]+)%/g)].map((m) => parseFloat(m[1]));

  const headToHead = readBars();
  checks["bar groups rendered"] = headToHead.length >= 4;
  for (const [i, group] of headToHead.entries()) {
    const widths = barWidths(group);
    const name = `bar group ${i}`;
    checks[`${name} has bars`] = widths.length === payload.runs.length;
    checks[`${name} worst bar fills the track`] = Math.max(...widths) > 99.9;
    checks[`${name} shortest bar is not empty`] = Math.min(...widths) > 0;
    // Every quantity is lower-is-better, so the group must run best to worst.
    const sorted = widths.every((w, j) => j === 0 || w >= widths[j - 1] - 1e-9);
    checks[`${name} runs best first`] = sorted;
    checks[`${name} states its scale`] = /full bar = /.test(group.html);
  }

  // A method whose routes SUMO could not drive has no measurement.  It must
  // disappear from the groups rather than draw a zero-width bar, which is
  // indistinguishable from a perfect result at a glance.
  const keep = state.data.runs[0].simulation;
  state.data.runs[0].simulation = null;
  sandbox.__renderHeadToHead();
  const withoutOne = readBars();
  // Only the SUMO-derived groups can lose the method; the objective comes from
  // the optimiser and is present for every run regardless.
  const measuredGroups = withoutOne.slice(1);
  checks["unmeasured method is dropped, not zeroed"] = measuredGroups.every(
    (g) => barWidths(g).length === payload.runs.length - 1
  );
  checks["objective group keeps every method"] = barWidths(withoutOne[0]).length === payload.runs.length;
  checks["dropped method is announced"] = measuredGroups.every((g) => /1 not measured/.test(g.html));
  checks["no zero-width bar anywhere"] = withoutOne.every((g) => Math.min(...barWidths(g)) > 0);
  state.data.runs[0].simulation = keep;
  sandbox.__renderHeadToHead();

  // Nothing anywhere may render as NaN or undefined: a value that failed to
  // serialise becomes the literal text of the cell.
  const renderedText = [
    elements.get("bars").innerHTML,
    elements.get("table").innerHTML,
    elements.get("measured").innerHTML,
    elements.get("map-sub").textContent,
  ].join(" ");
  checks["no NaN rendered"] = !renderedText.includes("NaN");
  checks["no undefined rendered"] = !renderedText.includes("undefined");
  checks["no Infinity rendered"] = !renderedText.includes("Infinity");

  // The per-vehicle bars are scaled to the slowest vehicle in the set, with the
  // fleet mean marked, so a single slow vehicle is visible against the others.
  sandbox.__select(state.data.runs[0].key);
  const measured = elements.get("measured").innerHTML;
  checks["per-vehicle bars scale to the slowest"] = /full bar = \d+ min, the slowest/.test(measured);
  checks["per-vehicle mean is marked"] = measured.includes("bar-tick");
  // Behavioural, not textual: the caption claiming "the slowest" proves nothing
  // if the widths are computed against some other reference.  The slowest
  // vehicle is in the set, so its bar has to fill the track exactly.
  const fleetWidths = [
    ...measured.matchAll(/class="bar-fill" style="width:([0-9.e-]+)%/g),
  ].map((m) => parseFloat(m[1]));
  checks["slowest vehicle fills the track"] =
    fleetWidths.length > 0 && Math.max(...fleetWidths) > 99.9;
  const tick = measured.match(/class="bar-tick" style="left:([0-9.e-]+)%/);
  const mean = state.data.runs[0].simulation.mean_travel_time_s;
  const slowest = Math.max(
    ...Object.values(state.data.runs[0].simulation.per_vehicle).map((v) => v.travel_time)
  );
  checks["mean tick sits at mean/slowest"] =
    Boolean(tick) && Math.abs(parseFloat(tick[1]) - (mean / slowest) * 100) < 0.5;
  checks["per-vehicle worst first"] = (() => {
    const values = [...measured.matchAll(/<span class="bar-val">(\d+) min<\/span>/g)].map((m) =>
      parseInt(m[1], 10)
    );
    return values.every((v, j) => j === 0 || v <= values[j - 1]);
  })();

  // Select every method in turn, so each card, row and route set is drawn.  This
  // reports only its own failures: folding the whole check set into one boolean
  // would report "a method is not selectable" whenever anything else failed,
  // sending the reader after the wrong bug.
  const selectFailures = [];
  for (const run of payload.runs) {
    try {
      sandbox.__select(run.key);
    } catch (err) {
      selectFailures.push(run.key + ": " + err);
    }
  }
  checks["every method selectable"] = selectFailures.length === 0;
  if (selectFailures.length) checks["select failures"] = selectFailures.join(" | ");

  // Drive the animation across and past the whole run.  A fault in the vehicle
  // loop only surfaces once a vehicle is actually on screen, and an undefined
  // palette entry throws from inside a canvas call where it reads like a no-op
  // stub rather than a bug.
  for (const id of ["show-base", "show-points", "show-anim"]) elements.get(id).checked = true;
  let drew = 0;
  elements.get("map").getContext = () => {
    const ctx = context2d();
    ctx.arc = () => { drew += 1; };
    return ctx;
  };
  for (const t of [0, 300, 600, 3000, 6000, 10800, 20000]) {
    state.frame = t;
    try {
      drew = 0;
      sandbox.__draw();
      checks["draws at t=" + t] = drew > 0;
    } catch (err) {
      checks["draws at t=" + t] = false;
      checks["draw at t=" + t + " error"] = String(err);
    }
  }

  const names = ["cargoTruck_3", "deliveryVan_0", "autoRickshaw_11", "noNumberHere"];
  const colours = names.map(sandbox.__vehicleColour);
  checks["vehicle colour always defined"] = colours.every(
    (c) => typeof c === "string" && c.startsWith("#")
  );
  checks["vehicle colour stable"] =
    sandbox.__vehicleColour("cargoTruck_3") === sandbox.__vehicleColour("cargoTruck_3");
  checks["different vehicles differ"] =
    sandbox.__vehicleColour("deliveryVan_0") !== sandbox.__vehicleColour("cargoTruck_1");

  // ------------------------------------------------- where each vehicle is going
  //
  // A card is only useful if its two labels are stops the vehicle really visits,
  // in an order the route actually prescribes, and the bar agrees with both.  The
  // strongest available check is the route itself: a label drawn from a different
  // vehicle's stop list, or from a stop the vehicle has not reached yet, is caught
  // by comparing the card against `waypoints[v]` rather than by looking for junk
  // characters in the markup.
  const legsHost = elements.get("fleet-legs");
  const readLegs = () => {
    const cards = [...legsHost.innerHTML.matchAll(
      /<div class="leg" style="--v:(#[0-9a-f]{6})">([\s\S]*?)<\/div><\/div>/g
    )];
    return cards.map(([, colour, body]) => ({
      colour,
      name: (body.match(/class="leg-name">([^<]*)</) || [])[1],
      count: (body.match(/class="leg-count">([^<]*)</) || [])[1],
      from: (body.match(/class="leg-node">([^<]*)</) || [])[1],
      to: (body.match(/class="leg-node">([^<]*)</g) || []).slice(1)
        .map((s) => s.replace(/.*>([^<]*)</, "$1"))[0],
      speed: (body.match(/class="leg-speed">([^<]*)</) || [])[1],
      bar: parseFloat((body.match(/class="leg-bar"><i style="width:([0-9.e-]+)%"/) || [])[1]),
    }));
  };

  const legProblems = [];
  for (const run of payload.runs) {
    sandbox.__select(run.key);
    const cards = readLegs();
    if (!run.simulation || !run.simulation.trace) {
      if (cards.length) legProblems.push(run.key + ": cards drawn for an un-simulated run");
      if (!/not simulated/.test(legsHost.innerHTML)) {
        legProblems.push(run.key + ": un-simulated run is not explained");
      }
      continue;
    }
    if (cards.length !== run.polylines.length) {
      legProblems.push(
        run.key + ": " + cards.length + " cards for " + run.polylines.length + " routes"
      );
    }
    for (const [i, card] of cards.entries()) {
      const v = Number(card.name.match(/(\d+)\s*$/)?.[1] ?? NaN);
      const marks = run.waypoints?.[v];
      if (!marks || !marks.length) {
        legProblems.push(run.key + " " + card.name + ": no stop sequence for this vehicle");
        continue;
      }
      // Both labels must be stops of *this* vehicle's route, and they must be
      // *consecutive* on it.  Membership alone is not enough: printing the stop a
      // vehicle just left under both headings names two stops that are both real
      // and both in the right order, and reads as a plausible, wrong answer.
      const pairs = marks
        .map((m, i) => (i + 1 < marks.length ? [m.label, marks[i + 1].label] : null))
        .filter(Boolean);
      const consecutive = pairs.filter(([a, b]) => a === card.from && b === card.to).length;
      if (consecutive !== 1) {
        legProblems.push(
          run.key + " " + card.name + ": " + card.from + " -> " + card.to +
            " is not a consecutive pair on this route"
        );
      }
      if (card.colour !== sandbox.__vehicleColour(card.name)) {
        legProblems.push(run.key + " " + card.name + ": colour does not match the map");
      }
      if (!(card.bar >= 0 && card.bar <= 100)) {
        legProblems.push(run.key + " " + card.name + ": bar " + card.bar + " outside 0-100");
      }
      if (!/^\d+ km\/h$/.test(card.speed)) {
        legProblems.push(run.key + " " + card.name + ": speed " + card.speed);
      }
      const reach = Number((card.count.match(/(\d+)\/(\d+)/) || [])[1]);
      const total = Number((card.count.match(/(\d+)\/(\d+)/) || [])[2]);
      if (total !== marks.filter((m) => m.kind === "customer").length) {
        legProblems.push(run.key + " " + card.name + ": stop total " + total + " vs " +
          marks.filter((m) => m.kind === "customer").length + " on route");
      }
      if (reach > total) legProblems.push(run.key + " " + card.name + ": reached more stops than exist");
    }
  }
  checks["leg cards are correct for every method"] = legProblems.length === 0;
  if (legProblems.length) checks["leg card problems"] = legProblems.slice(0, 8).join(" | ");

  // The stop a card names has to advance through the route as time passes, and
  // never backwards.  Dwell at a stop is legitimate, so only direction is checked
  // — but a panel frozen on one stop passes a direction-only test, so each
  // vehicle must also be seen naming at least two different stops.
  const advance = [];
  const seenStops = new Map();
  for (const run of payload.runs) {
    if (!run.simulation || !run.simulation.trace) continue;
    sandbox.__select(run.key);
    const seen = new Map();
    for (let step = 0; step <= 20; step += 1) {
      state.frame = (run.simulation.trace.steps * step) / 20;
      sandbox.__renderLegs(run);
      for (const card of readLegs()) {
        const v = Number(card.name.match(/(\d+)\s*$/)?.[1] ?? NaN);
        const marks = run.waypoints[v];
        const idx = marks.findIndex((m) => m.label === card.to);
        const prev = seen.get(card.name);
        // "Depot" is both the first and the last waypoint; the final one is the
        // later reading, so a jump to it is forward progress, not a wrap-around.
        if (prev !== undefined && idx < prev) {
          advance.push(run.key + " " + card.name + ": next stop went " + prev.label + " -> " + card.to);
        }
        seen.set(card.name, { idx, label: card.to });
        if (!seenStops.has(run.key + " " + card.name)) seenStops.set(run.key + " " + card.name, new Set());
        seenStops.get(run.key + " " + card.name).add(card.to);
      }
    }
  }
  checks["next stop only advances"] = advance.length === 0;
  if (advance.length) checks["next stop regressions"] = advance.slice(0, 5).join(" | ");
  const frozen = [...seenStops].filter(([who, stops]) => {
    const v = Number(who.split(" ").pop());
    const run = payload.runs.find((r) => r.key === who.split(" ")[0]);
    return stops.size < 2 && (run?.waypoints?.[v] || []).filter((m) => m.kind === "customer").length >= 2;
  });
  checks["every vehicle names more than one stop over the run"] = frozen.length === 0;
  if (frozen.length) checks["vehicles frozen on one stop"] = frozen.map(([w]) => w).slice(0, 5).join(" | ");

  // Hand every rendered container to the Python side: the CSS is only checkable
  // against real markup, and several of the worst defects here are invisible to
  // a DOM stub because it has no box model at all.  The text goes over too,
  // because what the reader actually reads is the text and not the markup.
  const rendered = { html: {}, text: {} };
  for (const [id, el] of elements) {
    rendered.html[id] = el.markup();
    rendered.text[id] = el.textContent || "";
  }

  const failed = Object.entries(checks).filter(([, v]) => v !== true);
  for (const [name, value] of failed) console.log("FAIL", name, value);
  require("fs").writeFileSync(process.argv[5], JSON.stringify(rendered));
  console.log(failed.length ? "demo UI check FAILED" : "demo UI check passed: " + Object.keys(checks).length + " checks");
  process.exit(failed.length ? 1 : 0);
}, 400);
"""


BLOCK_LEVEL = re.compile(
    r"display\s*:\s*(block|flow-root|flex|grid|inline-block|inline-flex|list-item|table)\b"
)
INLINE_TAGS = {"span", "i", "b", "em", "strong", "a", "label", "small", "sub", "sup", "abbr"}


def css_declarations(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """Rule bodies keyed by class name, and separately by bare tag name."""
    classes: dict[str, str] = {}
    tags: dict[str, str] = {}
    for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        for name in re.findall(r"\.([A-Za-z][\w-]*)", selector):
            classes[name] = classes.get(name, "") + body
        for name in re.findall(r"(?:^|[\s>+~])([a-z][a-z0-9]*)(?![\w-]*\s*[.:#\[])", selector):
            tags[name] = tags.get(name, "") + body
    return classes, tags


def check_sized_inline_elements(css: str, rendered: dict[str, str]) -> list[str]:
    """An element sized by a width must be a box, not an inline run of text.

    ``width`` and ``height`` do not apply to a non-replaced inline element, so a
    ``<span class="bar-fill" style="width:82%">`` lays out at 0x0 and renders
    nothing at all.  Every value on the page is still correct and every string
    assertion still passes, because the defect is entirely in layout — which is
    why it needs its own check rather than being left to a visual review.

    The rule that gives the element its box may be keyed on a class or on the tag
    itself, so both are consulted; ``.leg-bar > i`` is sized by a bare tag
    selector and would be invisible to a class-only lookup.
    """
    classes, tags = css_declarations(css)
    problems: list[str] = []
    for where, html in rendered.items():
        for name, attrs in re.findall(r"<([a-zA-Z][\w-]*)\b([^>]*)>", html):
            if name.lower() not in INLINE_TAGS:
                continue
            sized = re.search(r"(width|height)\s*:", attrs)
            if not sized:
                continue
            names = [one for cls in re.findall(r'class="([^"]+)"', attrs) for one in cls.split()]
            selector = "." + ".".join(names) if names else name.lower()
            bodies = (
                [classes.get(n, "") for n in names] if names else [tags.get(name.lower(), "")]
            )
            for body in bodies:
                if body and not BLOCK_LEVEL.search(body):
                    problems.append(
                        f"#{where}: <{name} class=\"{' '.join(names)}\"> carries a "
                        f"{sized.group(1)} but {selector} never sets a block-level display"
                    )
    return problems


def element_ids(html: str) -> list[str]:
    return re.findall(r'id="([^"]+)"', html)


EM_DASH = "\u2014"
CANVAS_TAG = re.compile(r"<canvas\b([^>]*)>", re.IGNORECASE)


def check_canvases_are_named(html: str) -> list[str]:
    """A canvas that carries information has to say what it is.

    Both canvases on this page hold the only copy of their content: the routes
    and the convergence curves are pixels, and there is no text alternative
    anywhere near them.  A screen reader reaches a bare ``<canvas>`` and finds
    nothing to announce, so the two most important things on the page are simply
    absent.  ``role="img"`` plus a name is what makes them reachable, and the
    name is taken from the visible heading so it cannot drift out of date.
    """
    ids = set(element_ids(html))
    problems: list[str] = []
    for attrs in CANVAS_TAG.findall(html):
        found = re.search(r'id="([^"]+)"', attrs)
        name = f"#{found.group(1)}" if found else "<canvas with no id>"
        if not re.search(r'role="img"', attrs):
            problems.append(f'{name}: no role="img", so it is not exposed as an image')
        label = re.search(r'\baria-label(?:ledby)?="([^"]*)"', attrs)
        if not label or not label.group(1).strip():
            problems.append(f"{name}: no accessible name (aria-label or aria-labelledby)")
            continue
        attribute = "aria-label" if "aria-label=" in attrs else "aria-labelledby"
        for target in label.group(1).split():
            if target not in ids:
                problems.append(f"{name}: {attribute} points at {target!r}, which is not an id in index.html")
        described = re.search(r'\baria-describedby="([^"]*)"', attrs)
        for target in described.group(1).split() if described else []:
            if target not in ids:
                problems.append(f"{name}: aria-describedby points at {target!r}, which is not an id in index.html")
    if not CANVAS_TAG.search(html):
        problems.append("no <canvas> found; the markup was probably restructured past this check")
    return problems


def check_no_implicit_requests(html: str) -> list[str]:
    """Declare the favicon, or the browser will ask for one and be told no.

    A page with no ``<link rel="icon">`` makes the browser request
    ``/favicon.ico`` on every load.  The server has no such file, so the console
    opens with a red 404 that has nothing to do with the page being correct,
    which is a poor first impression for a demonstration.
    """
    if re.search(r'<link\b[^>]*\brel="[^"]*\bicon\b', html, re.IGNORECASE):
        return []
    return ["no <link rel=\"icon\">: the browser will request /favicon.ico and log a 404"]


TEXT_BEARING_INPUTS = {"", "text", "search", "email", "url", "tel", "password", "number", "submit", "reset", "button"}


def check_form_controls_are_legible(markup: list[tuple[str, str]], css: str) -> list[str]:
    """A form control that shows words has to be told what colour they are.

    ``color`` is one of the properties that does not inherit into a replaced
    element: the user agent stylesheet gives every ``<button>`` its own
    ``buttontext`` colour, black, and that beats anything inherited from an
    ancestor.  On a dark panel a button whose rule never names a colour is
    therefore black on near-black.  That is not hypothetical: the method name
    and the objective on all nine method cards rendered at a contrast ratio of
    1.33, which is why the base ``button`` rule now carries a colour.

    Neither the rendered-string assertions nor this harness can see it, because
    the colour is chosen by the user agent rather than by the stylesheet, and the
    DOM stub has no cascade at all.  Only the stylesheet and the markup, read
    together, can.

    The markup has to be both the static file and what JavaScript rendered, or
    the check is blind to exactly the case that matters: the method cards exist
    only in the rendered output, so a check over index.html alone passes on a
    page whose most prominent text is invisible.

    Controls that paint no text of their own are exempt: a checkbox and a range
    slider take their colour from the theme without it mattering.
    """
    classes, tags = css_declarations(css)
    problems: list[str] = []
    # Nine identical cards produce nine identical sentences, so count them and
    # report the pattern once: the fix is one line of CSS, not nine edits.
    tally: dict[tuple[str, str, tuple[str, ...]], int] = {}
    for where, html in markup:
        for name, attrs in re.findall(r"<([a-zA-Z][\w-]*)\b([^>]*)>", html):
            tag = name.lower()
            if tag not in {"button", "select", "textarea", "input"}:
                continue
            if tag == "input":
                kind = (re.search(r'type="([^"]*)"', attrs) or [None, ""])[1].lower()
                if kind not in TEXT_BEARING_INPUTS:
                    continue
            found = re.search(r'id="([^"]+)"', attrs)
            named = [c for group in re.findall(r'class="([^"]+)"', attrs) for c in group.split()]
            bodies = [classes.get(c, "") for c in named] + [tags.get(tag, "")]
            if not any(re.search(r"(?:^|;)\s*color\s*:", body) for body in bodies if body):
                label = f"#{found.group(1)}" if found else f"<{tag}>"
                tally[(where, label, tuple(named))] = tally.get((where, label, tuple(named)), 0) + 1
    for (where, label, named), count in tally.items():
        times = f", {count} elements like it" if count > 1 else ""
        problems.append(
            f'{where} {label} (class "{" ".join(named)}"){times}: no rule sets color, so it takes '
            f"the user agent's buttontext, which is black on this panel"
        )
    return problems


def check_no_em_dash(html: str, rendered: dict[str, str]) -> list[str]:
    """No em dash in anything the reader sees.

    The house style bans them, and they are easy to reintroduce by accident
    because a dash reads naturally in a heading and nowhere in the code announces
    the rule.  The rendered markup is scanned as well as the rendered text: this
    harness's DOM stub stores ``innerHTML`` without ever computing
    ``textContent``, so a heading written from JavaScript is only ever visible in
    the markup, and a text-only check would pass while the page showed a dash.
    Source comments are not scanned, because a dash in a comment reaches nobody.
    """
    problems: list[str] = []
    for number, line in enumerate(html.splitlines(), 1):
        if EM_DASH in line:
            problems.append(f"index.html:{number}: {line.strip()[:72]}")
    for kind, group in rendered.items():
        for where, markup in group.items():
            if EM_DASH in (markup or ""):
                problems.append(f"#{where} ({kind}): {markup.strip()[:72]}")
    return problems


def main() -> int:
    payload = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/demo/comparison.json")
    if not payload.is_file():
        print(f"no comparison payload at {payload}; build one with `qitransit demo --build-only`")
        return 1

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    source = (STATIC / "app.js").read_text(encoding="utf-8")
    ids = set(element_ids(html))
    referenced = set(re.findall(r'\$\("([^"]+)"\)', source))
    # An id reached only by ARIA is still used: that is how a canvas is given the
    # name of the heading above it, and app.js never has to touch it.  Only the
    # id-list attributes count; aria-label holds prose, and splitting it on
    # spaces would report every word of it as a missing element id.
    referenced |= set(re.findall(r'\baria-(?:labelledby|describedby|controls)="([^"]*)"', html))
    referenced = {token for group in referenced for token in group.split()}
    orphans = referenced - ids
    if orphans:
        print(f"referenced element ids absent from index.html: {sorted(orphans)}")
        return 1

    unused = ids - referenced
    if unused:
        print(f"note: ids in index.html no longer used by app.js or ARIA: {sorted(unused)}")

    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(HARNESS)
        harness_path = handle.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as dump:
        dump_path = dump.name

    result = subprocess.run(
        [
            "node",
            harness_path,
            str(payload),
            str(STATIC / "index.html"),
            str(STATIC / "app.js"),
            dump_path,
        ],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    Path(harness_path).unlink(missing_ok=True)

    failures = result.returncode

    naming = check_canvases_are_named(html)
    for problem in naming:
        print("FAIL canvas accessible name:", problem)
    if naming:
        failures = 1
    else:
        print("check passed: every canvas is exposed as a named image")

    implicit = check_no_implicit_requests(html)
    for problem in implicit:
        print("FAIL implicit request:", problem)
    if implicit:
        failures = 1
    else:
        print("check passed: the page declares its own favicon")

    if dump_path and Path(dump_path).stat().st_size:
        rendered = json.loads(Path(dump_path).read_text(encoding="utf-8"))

        legible = check_form_controls_are_legible(
            [("index.html", html), *sorted(rendered["html"].items())],
            (STATIC / "style.css").read_text(encoding="utf-8"),
        )
        for problem in legible:
            print("FAIL form control colour:", problem)
        if legible:
            failures = 1
        else:
            print("check passed: every text-bearing form control is given a colour")

        problems = check_sized_inline_elements(
            (STATIC / "style.css").read_text(encoding="utf-8"), rendered["html"]
        )
        for problem in problems:
            print("FAIL sized inline element:", problem)
        if problems:
            failures = 1
        else:
            print("check passed: no inline element is sized by width or height")

        dashes = check_no_em_dash(html, rendered)
        for problem in dashes:
            print("FAIL em dash in reader-facing text:", problem)
        if dashes:
            failures = 1
        else:
            print("check passed: no em dash in reader-facing text")
    Path(dump_path).unlink(missing_ok=True)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
