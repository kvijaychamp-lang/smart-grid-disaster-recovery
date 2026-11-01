# ⚡ Smart Grid Disaster Recovery & Rerouting Optimization System

> **B.Tech Computer Science & Engineering — Discrete Mathematics Project**
> Academic Year 2026-27 | Python · Streamlit · NetworkX

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Core Discrete Mathematics Modules](#-core-discrete-mathematics-modules)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Local Installation & Setup](#-local-installation--setup)
6. [Running the Dashboard](#-running-the-dashboard)
7. [Running Tests](#-running-tests)
8. [Team & Contributions](#-team--contributions)

---

## 🎯 Project Overview

Modern electrical smart grids are large-scale, interconnected networks
susceptible to cascading failures caused by natural disasters, cyberattacks,
or equipment faults. This system applies **Discrete Mathematics** concepts
to:

- **Model** a regional smart grid as a weighted graph.
- **Detect** critical vulnerabilities (cut-vertices and bridges) that, if
  removed, would disconnect the network.
- **Recover** the minimum-cost spanning backbone using **Kruskal's MST**.
- **Reroute** power flow along shortest safe paths using **Dijkstra's
  algorithm**.
- **Prioritise** substation restoration using **POSET** (Partially Ordered
  Set) scheduling theory.
- **Visualise** all analyses interactively through a Streamlit dashboard.

---

## 🧮 Core Discrete Mathematics Modules

### 1. Graph Theory — Grid Modelling
| Concept | Application |
|---|---|
| Vertices (nodes) | Substations, power plants, distribution hubs |
| Edges (links) | Transmission lines between nodes |
| Edge weights | Line capacity (MW), distance (km), or fault cost |
| Directed / Undirected | Mixed-mode grid segments |

The grid is represented as a weighted undirected graph **G = (V, E, w)** using
NetworkX. Disaster events remove a subset of nodes/edges, producing a
disconnected subgraph **G' ⊆ G**.

---

### 2. Cut-Sets — Vulnerability Analysis
A **cut-set** is a minimal set of edges whose removal disconnects the graph.
Special cases:
- **Cut-Vertex (Articulation Point):** A node whose removal increases the
  number of connected components — found via Tarjan's DFS algorithm in **O(V+E)**.
- **Bridge (Cut-Edge):** An edge whose removal disconnects the graph.

> Identifying cut-vertices and bridges reveals the most critical
> infrastructure that must be protected or restored first.

---

### 3. POSET — Recovery Priority Scheduling
A **Partially Ordered Set (POSET)** `(S, ≤)` models precedence constraints
among restoration tasks:
- Substations with higher topological importance are ordered before
  downstream loads.
- The Hasse diagram visualises the priority DAG.
- Topological sort of the POSET yields the optimal restoration sequence.

---

### 4. Kruskal's MST — Recovery Backbone
**Kruskal's Minimum Spanning Tree** algorithm constructs the cheapest
connected subgraph of the surviving grid:
- Sort all surviving edges by weight.
- Greedily add edges using a **Union-Find (Disjoint Set Union)** structure.
- Result: minimum-cost backbone to restore basic connectivity — **O(E log E)**.

---

### 5. Dijkstra's Algorithm — Optimal Rerouting
**Dijkstra's Single-Source Shortest Path** finds the lowest-cost rerouting
paths from any source substation to all destinations:
- Uses a min-heap (priority queue) — **O((V + E) log V)**.
- Applied post-MST to optimise actual power-flow routing on the recovered grid.

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.12+ | Core implementation |
| Dashboard | Streamlit 1.64 | Interactive web UI |
| Graph Engine | NetworkX 3.7 | Graph construction & algorithms |
| Visualisation | Matplotlib 3.11 | Static graph plots |
| Interactive Viz | Plotly 7.0 | Dynamic, interactive charts |
| Data Handling | Pandas 3.1 | Grid topology / scenario data |
| Numerical | NumPy 2.5 | Matrix ops, weight calculations |
| Testing | pytest 9.2 | Unit & integration test suite |

---

## 📁 Project Structure

```
smart-grid-disaster-recovery/
│
├── requirements.txt              # Pinned Python dependencies
├── conftest.py                   # pytest shared fixtures
├── .gitignore
│
├── src/                          # Core library (importable package)
│   ├── __init__.py
│   ├── graph_engine/
│   │   ├── grid_graph.py         # Graph construction from topology configs
│   │   └── grid_loader.py        # JSON/CSV topology loader
│   ├── algorithms/
│   │   ├── dijkstra.py           # Shortest-path rerouting
│   │   ├── kruskal_mst.py        # Minimum spanning tree recovery
│   │   └── bfs_dfs.py            # Connectivity traversals
│   ├── poset/
│   │   ├── poset_model.py        # POSET construction & topological sort
│   │   └── hasse_diagram.py      # Hasse diagram renderer
│   ├── cutset/
│   │   ├── cut_vertex.py         # Articulation point detection (Tarjan)
│   │   └── bridge_finder.py      # Bridge / cut-edge detection
│   ├── simulation/
│   │   ├── disaster_simulator.py # Fault injection engine
│   │   └── recovery_planner.py   # End-to-end recovery orchestrator
│   └── utils/
│       ├── graph_utils.py        # Metrics, export, validation helpers
│       └── viz_utils.py          # Shared Matplotlib / Plotly helpers
│
├── dashboard/                    # Streamlit multi-page application
│   ├── __init__.py
│   ├── app.py                    # Entry-point: streamlit run dashboard/app.py
│   ├── pages/
│   │   ├── 01_grid_topology.py   # Grid viewer
│   │   ├── 02_disaster_sim.py    # Disaster simulation controls
│   │   ├── 03_mst_recovery.py    # Kruskal MST visualisation
│   │   ├── 04_rerouting.py       # Dijkstra rerouting paths
│   │   └── 05_poset_analysis.py  # POSET priority & Hasse diagram
│   └── components/
│       ├── graph_plot.py         # Reusable graph rendering component
│       └── metrics_panel.py      # Sidebar metrics component
│
├── tests/                        # pytest test suite
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_dijkstra.py
│   │   ├── test_kruskal.py
│   │   ├── test_cutset.py
│   │   └── test_poset.py
│   └── integration/
│       └── test_recovery_pipeline.py
│
├── data/                         # Sample grid data & scenarios
│   ├── grid_topologies/
│   │   └── ieee_14bus.json       # IEEE 14-bus benchmark topology
│   └── disaster_scenarios/
│       └── scenario_flood.json   # Example flood-event fault mask
│
├── notebooks/                    # Jupyter exploration notebooks
│   ├── 01_graph_theory_demo.ipynb
│   └── 02_mst_analysis.ipynb
│
├── docs/                         # Reports & diagrams
│   ├── diagrams/
│   └── reports/
│
└── assets/                       # Static UI assets
    ├── images/
    └── styles/
```

---

## 💻 Local Installation & Setup

### Prerequisites
- Python **3.12** or higher ([download](https://www.python.org/downloads/))
- `pip` package manager (bundled with Python)
- Git (optional, for cloning)

### Step 1 — Clone or Download the Repository
```bash
git clone https://github.com/<your-org>/smart-grid-disaster-recovery.git
cd smart-grid-disaster-recovery
```

### Step 2 — Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Verify Installation
```bash
python -c "import streamlit, networkx, matplotlib, plotly, pandas, numpy; print('All dependencies OK ✔')"
```

---

## 🚀 Running the Dashboard

```bash
# Run the unified Control Center dashboard (Recommended)
streamlit run app.py

# Alternatively, run the multi-page dashboard
streamlit run dashboard/app.py
```

The dashboard will open automatically at **http://localhost:8501**.

---

## 🧪 Running Tests & Validation

```bash
# Run the core 78-test evaluation suite
pytest test_grid.py -v

# Run the complete test suite (152 tests across unit, integration, and performance)
pytest -v

# Run the fast standalone smoke test
python smoke_test.py
```

---

## 👥 Team & Contributions

| Role | Responsibility |
|---|---|
| Graph Engine Lead | Grid modelling, NetworkX integration |
| Algorithm Developer | Dijkstra, Kruskal, Tarjan implementations |
| POSET & Theory Lead | POSET model, Hasse diagram, proof write-up |
| Dashboard Developer | Streamlit UI, Plotly visualisations |
| QA / Testing Lead | pytest suite, scenario validation |

---

> **Academic Integrity Notice:** This project is developed for educational
> purposes as part of the B.Tech CSE curriculum. All algorithmic
> implementations are original unless otherwise cited.

---

*Last Updated: October 2026*
