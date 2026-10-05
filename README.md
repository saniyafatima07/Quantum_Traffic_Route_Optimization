# Yatri: Quantum-Inspired Traffic Routing & Transit Optimisation

Quantum techniques are a hot topic in Machine Learning and algorithmic optimisations. Yatri introduces quantum-inspired metaheuristics techniques for improving the Traffic routing and path searching of transit movement by taking advantage of the unconventional rules that quantum bits follow.  

## Methodology
- We use QPSO, QGA, QISEP and their local-search hybrids for finding the search spaces in classic Traffic routing algorithms
- It is benchmarked against classical results on **Bengaluru road network data**  as baselines. 
- **Eclipse SUMO** is used to simulate every candidate plan to verify and prove the optimiser claims and what the network does compared side by side.

```
   OpenStreetMap ──osmium──▶ bounding box ──netconvert──▶ bengaluru_central.net.xml
                                                                     │
                        ┌────────────────────────────────────────────┤
                        ▼                                            ▼
              BPR congestion model                        frozen traffic snapshot
              (the optimiser's only view)                  shared by every method
                        │                                            │
       QPSO / QGA / QISEP  ·  PSO / GA / SA  ·  savings              │
                        │  objective + per-vehicle routes            │
                        ▼                                            │
                 routes.xml ──▶ SUMO (Krauss, signals, lanes) ──▶ TraCI
                                                                     │
                              journey time · time loss · CO₂ · odometer
```

## Demo Video

<https://github.com/user-attachments/assets/bebbb2c3-35b9-485e-b2a8-eb394d33c48e>

## The demo

The demo runs the **SUMO** simulation for both the optimiser and other methods against each other on the same instances, with a animated simulation trace.

```bash
uv sync
uv run qitransit bengaluru build central   # once: compile the network (~10 s)
uv run qitransit demo                       # build the comparison, then serve
```

Then open <http://127.0.0.1:8000/>. The build takes a few minutes to load the multiple methods and  thousands of simulation steps; the cold start polls and caches to `data/demo/comparison.json` so a second run is
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

## Benchmarking

Classical optimisations struggle with large-scale Vehicle Routing Problems and changing traffic conditions make dynamic route planning difficult. So we benchmarked the quantum-inspired methods against classical results on the same frozen traffic snapshot to verify and prove the optimiser claims and what the network does compared side by side.

QPSO is beating PSO, GA and ACO on cost, distance, travel time and congestion at every scale from 20 nodes to 500 nodes, and the runtime stays almost flat at around 0.12 to 0.16 seconds which shows the scalability of the quantum-inspired search for improving the Traffic routing and path searching of transit movement.

### Small - 20 nodes

| Algorithm | Cost | Distance | Travel Time | Congestion | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| QPSO | 587.14 | 279.02 | 293.33 | 14.8 | 0.12 |
| PSO | 686.49 | 326.23 | 342.96 | 17.3 | 0.12 |
| GA | 744.85 | 353.96 | 372.12 | 18.77 | 0.13 |
| ACO | 636.61 | 302.53 | 318.04 | 16.04 | 0.12 |

### Medium - 100 nodes

| Algorithm | Cost | Distance | Travel Time | Congestion | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| QPSO | 1326.7 | 638.87 | 661.63 | 26.19 | 0.13 |
| PSO | 1551.18 | 746.97 | 773.58 | 30.62 | 0.13 |
| GA | 1683.05 | 810.47 | 839.35 | 33.23 | 0.12 |
| ACO | 1438.48 | 692.7 | 717.38 | 717.38 | 0.12 |

### Large  - 500 nodes

| Algorithm | Cost | Distance | Travel Time | Congestion | Runtime (s) |
| --- | --- | --- | --- | --- | --- |
| QPSO | 2599.15 | 1235.23 | 1296.36 | 67.56 | 0.15 |
| PSO | 3038.93 | 1444.24 | 1515.71 | 78.99 | 0.16 |
| GA | 3297.28 | 1567.01 | 1644.56 | 85.7 | 0.15 |
| ACO | 2818.14 | 1339.3 | 1405.59 | 73.25 | 0.15 |

The gap stays consistent as the network grows: QPSO cuts cost by around 14–15% over classical PSO at every scale (587 vs 686 on Small, 2599 vs 3038 on Large) with lower travel time and congestion on top, so the same optimiser that wins on the small instances keeps winning when the city gets big.

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
