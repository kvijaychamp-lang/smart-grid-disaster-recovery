"""
engine.py — Root-level Proxy for Smart Grid Discrete Mathematics Engine
========================================================================
B.Tech CSE — Discrete Mathematics Project  |  Academic Year 2026-27

This module proxies all exports from `src.engine` to ensure clean, seamless
imports across both root-level scripts (`app.py`, `smoke_test.py`, `test_grid.py`)
and package modules (`from src.engine import ...` or `import engine`).

Mathematical Modules Proxied
----------------------------
  § 1  Graph Theory         G = (V, E, w) grid construction
  § 2  Cut-Set Analysis     Tarjan's DFS bridge & cut-vertex detection O(V + E)
  § 3  POSET Scheduling     Lattice partial-order priority allocation O(V log V)
  § 4  Dijkstra SSSP        Minimum-resistance shortest paths O((V + E) log V)
  § 5  Kruskal MST          Minimum spanning tree/forest via Union-Find O(E log E)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable
_SRC_DIR = Path(__file__).parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Re-export all public API members from src.engine
from src.engine import (
    # Core Data Models & Enums
    NodeType,
    PriorityLevel,
    CutSetResult,
    POSETSchedule,
    DijkstraResult,
    KruskalResult,
    AuditEvent,
    ScenarioMetrics,
    # Edge Attribute Constants
    EDGE_RESISTANCE,
    EDGE_CAPACITY_MW,
    EDGE_POWER_LOSS,
    EDGE_VOLTAGE_KV,
    EDGE_STATUS,
    EDGE_LENGTH_KM,
    # Core Mathematical Functions
    initialize_grid_graph,
    detect_cut_sets,
    compute_poset_schedule,
    POSET_Priority_Scheduler,
    poset_priority_scheduler,
    dijkstra_shortest_path,
    dijkstra,
    Dijkstra,
    kruskal_mst,
    run_full_recovery_pipeline,
    # Query & Transformation Helpers
    apply_disaster,
    find_generator_nodes,
    find_critical_nodes,
    get_node_dataframe,
    get_edge_dataframe,
    grid_metrics,
    # SCADA Incident Audit & Scenario Evaluation
    create_audit_event,
    get_disaster_preset_config,
    evaluate_scenario,
    compare_scenario_metrics,
)

__all__ = [
    "NodeType",
    "PriorityLevel",
    "CutSetResult",
    "POSETSchedule",
    "DijkstraResult",
    "KruskalResult",
    "AuditEvent",
    "ScenarioMetrics",
    "EDGE_RESISTANCE",
    "EDGE_CAPACITY_MW",
    "EDGE_POWER_LOSS",
    "EDGE_VOLTAGE_KV",
    "EDGE_STATUS",
    "EDGE_LENGTH_KM",
    "initialize_grid_graph",
    "detect_cut_sets",
    "compute_poset_schedule",
    "POSET_Priority_Scheduler",
    "poset_priority_scheduler",
    "dijkstra_shortest_path",
    "dijkstra",
    "Dijkstra",
    "kruskal_mst",
    "run_full_recovery_pipeline",
    "apply_disaster",
    "find_generator_nodes",
    "find_critical_nodes",
    "get_node_dataframe",
    "get_edge_dataframe",
    "grid_metrics",
    "create_audit_event",
    "get_disaster_preset_config",
    "evaluate_scenario",
    "compare_scenario_metrics",
]

