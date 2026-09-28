# qitransit — Quantum-Inspired Traffic Routing & Transit Optimisation

Quantum-inspired metaheuristics (QPSO, QGA, QISEP) and their local-search
hybrids, benchmarked against classical baselines on **real Bengaluru road
network data**, with every candidate plan driven through **Eclipse SUMO** so
that what the optimiser claims and what the network actually does can be
compared side by side.

```
   OpenStreetMap ──osmium──▶ bounding box ──netconvert──▶ bengaluru_central.net.xml
                                                                     │
                        ┌────────────────────────────────────────────┤
                        ▼                                            ▼
              BPR congestion model                        frozen traffic snapshot
              (the optimiser's only view)                  shared by every method
                        │                                            │
       QPSO / QGA / QISEP  ·  PSO / GA / SA  ·  savings                │
                        │  objective + per-vehicle routes             │
                        ▼                                            │
                 routes.xml ──▶ SUMO (Krauss, signals, lanes) ──▶ TraCI
                                                                     │
                              journey time · time loss · CO₂ · odometer
```

## The demo

This is the part worth running. It serves a single page that puts the
optimiser's own numbers and SUMO's measurements next to each other for every
method, on the same instance, with the routes drawn over the city and the fleet
animated from the simulation trace.

```bash
uv sync
uv run qitransit bengaluru build central   # once: compile the network (~10 s)
uv run qitransit demo                       # build the comparison, then serve
```

Then open <http://127.0.0.1:8000/>. The build takes a few minutes (nine methods,
each simulated for 10 800 SUMO steps); the page polls for progress and fills in
as results land, and caches to `data/demo/comparison.json` so a second run is
instant.

Useful flags:

```bash
uv run qitransit demo --build-only              # build without serving
uv run qitransit demo --port 8811               # different port
uv run qitransit demo --methods qpso,sa,savings # a subset, for a quick look
uv run qitransit demo --customers 18 --budget 600 --steps 3600
```

`make demo`, `make demo-build`, `make demo-ui` and `make check` wrap the common
cases.

### What the page is careful about

- **Every objective re-scores its own routes.** A method's reported objective is
  recomputed from the exact edge sequence handed to SUMO, so the modelled and
  measured columns in a row always describe the same plan. `make check` asserts
  this for all nine methods, along with full service, route legality, and a
  plausible driven-to-priced distance ratio.
- **Time and distance are per vehicle in both column groups.** The objective is a
  fleet total — correct for ranking methods, wrong for reading against a
  per-vehicle mean — so the table never puts the two side by side unreconciled.
- **`model err %` is reported whether or not it flatters the model.** On the
  current instance it ranges from about 1 % to 48 %, and it is worst for the
  aggressive low-vehicle-count plans, which is the interesting part.

### Reading the result

On the default instance (29 customers, 3 000 evaluations, 5–6 vehicles) the
quantum-inspired methods reach a lower objective than the classical baselines,
and the hybrids reach the lowest of all. But the best objective does not
necessarily give the best realised run: the method that minimises the surrogate
most is not the one SUMO drives fastest. The page shows both, and the
disagreement is a finding rather than an error to be hidden.

## Command line

```bash
uv run qitransit --help
uv run qitransit methods                       # the registered optimisers
uv run qitransit net build manhattan           # synthetic city, for testing
uv run qitransit net info                      # statistics for a compiled net
uv run qitransit solve qisep-ls --budget 3000  # solve and print routes
uv run qitransit simulate data/demo/routes/qpso.rou.xml --steps 10800
uv run python scripts/validate_loop.py         # optimise then simulate, in one go
```

`SUMO_HOME` must point at a SUMO installation (`/usr/share/sumo` by default);
the TraCI bindings are picked up from `$SUMO_HOME/tools` automatically.

## What is implemented

| Layer | Module | Contents |
| --- | --- | --- |
| Network model | `qitransit.graph` | real Bengaluru extract, synthetic city generator, plain-XML emitter, `netconvert` driver, CSR adjacency, turn graph |
| Traffic model | `qitransit.routing.traffic` | BPR congestion cost, static snapshot, live TraCI cost |
| Shortest path | `qitransit.routing.shortest_path` | turn-graph Dijkstra, bidirectional Dijkstra, A\* with ALT landmarks, time-dependent search |
| Problems | `qitransit.problems` | congested shortest path, TSP, CVRP, VRPTW, travel matrix, feasibility repair |
| Quantum algorithms | `qitransit.algorithms.quantum` | QPSO, QGA, QISEP, each with a local-search hybrid |
| Classical baselines | `qitransit.algorithms.classical` | PSO, GA, SA, 2-opt/Or-opt, Clarke–Wright savings |
| Exact methods | `qitransit.algorithms.exact` | Held–Karp DP, exact shortest path, Dijkstra certification |
| Simulation | `qitransit.simulation` | SUMO session, route injection, departure scheduling, per-vehicle measurement, animated traces |
| Demo | `qitransit.demo` | the comparison page and its stdlib HTTP server |

## Notes on the modelling

A few decisions that are easy to get wrong, recorded so they are not "fixed"
later by accident:

- **Routing runs over the turn graph**, with the search state being the edge just
  traversed. A node-only shortest path can be optimal and still be undrivable,
  because two legal paths can meet at a junction with no permitted continuation
  between them. Legs are chained with `dijkstra_continuing` for the same reason.
- **ALT landmarks use one-sided gaps.** `|d(L,v) − d(L,t)|` is not admissible on
  a directed graph, so the bound is `max(d(L,t) − d(L,v), d(v,L) − d(t,L))`.
  A geometric heuristic was tried first and rejected: 2.7 % of the real
  network's edges violate the straight-line Lipschitz bound.
- **Aggregates cover only vehicles that finished.** A journey truncated at the
  simulation horizon is not a short journey.
- **SUMO's odometer includes junction-internal driving**, which no sum of edge
  lengths contains; the driven distance runs about 1.35× the routed edge length
  on this network. `getDistance` was verified against tripinfo `routeLength` to
  four decimal places.
- **Congestion is a frozen snapshot** shared by every method, so a difference
  between two results is attributable to the search and not to a different rush
  hour.

## Status

The demo, the nine optimisers, the real network pipeline and the SUMO
measurement layer are working and checked. Still to do: the statistical
benchmark harness in `qitransit.benchmark` (its runner has known bugs and the
package is missing its `__init__.py`), the `tests/` suite, and the write-ups
under `docs/`.
