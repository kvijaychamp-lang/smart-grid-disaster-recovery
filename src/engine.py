"""
engine.py — Smart Grid Disaster Recovery & Rerouting Optimization System
=========================================================================
Core Backend Module  |  B.Tech CSE — Discrete Mathematics Project

Mathematical Models Implemented
-------------------------------
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  § 1  Graph Theory        G = (V, E, w)  — weighted undirected graph    │
  │       Node types: Generator, Hospital, Emergency, Substation,           │
  │       Commercial, Residential  (6 NodeType constants)                    │
  │       Edge weight w(e) = resistance Ω  (Dijkstra & Kruskal primary key) │
  │                                                                          │
  │  § 2  Cut-Set Analysis    Tarjan's iterative DFS  — O(V + E)            │
  │       Bridges B ⊆ E : removal of e ∈ B disconnects G                   │
  │       Articulation Points A ⊆ V : removal of v ∈ A disconnects G       │
  │       Blacked-Out Zones: components with no generator source            │
  │                                                                          │
  │  § 3  POSET Scheduling    (S, ≤)  strict partial order on priority      │
  │       CRITICAL(1) ⊑ HIGH(2) ⊑ MEDIUM(3) ⊑ LOW(4) ⊑ OFFLINE(5)       │
  │       Hasse diagram covering relations  →  lattice_edges                 │
  │       Greedy resource allocation respects the total ordering             │
  │                                                                          │
  │  § 4  Dijkstra SSSP       O((V + E) log V)  binary min-heap            │
  │       weight_attr = 'resistance' (Ω)  —  lazy deletion variant          │
  │       Source: generator node  →  targets: all reachable nodes           │
  │                                                                          │
  │  § 5  Kruskal MST         O(E log E)  Union-Find with path compression  │
  │       Produces spanning forest on residual graph post-disaster           │
  │       Union-by-rank + path compression  →  O(α(V)) per operation        │
  └──────────────────────────────────────────────────────────────────────────┘

Author  : Smart Grid Project Team
Date    : 2026-10-31
Python  : 3.12+
Deps    : networkx>=3.7, numpy>=2.5, pandas>=3.1
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Generator

import networkx as nx
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# § 0  CONSTANTS, EXPORTS & ENUMERATIONS
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Data Models & Enums
    "NodeType",
    "PriorityLevel",
    "CutSetResult",
    "POSETSchedule",
    "DijkstraResult",
    "KruskalResult",
    "AuditEvent",
    "ScenarioMetrics",
    # Edge Attribute Constants
    "EDGE_RESISTANCE",
    "EDGE_CAPACITY_MW",
    "EDGE_POWER_LOSS",
    "EDGE_VOLTAGE_KV",
    "EDGE_STATUS",
    "EDGE_LENGTH_KM",
    # Core Mathematical Functions
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
    # Query, Audit & Scenario Helpers
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

class NodeType:
    """String constants for node types in the power grid."""
    GENERATOR   = "Generator"
    SUBSTATION  = "Substation"
    HOSPITAL    = "Hospital"
    RESIDENTIAL = "Residential"
    COMMERCIAL  = "Commercial"
    EMERGENCY   = "Emergency"


class PriorityLevel(IntEnum):
    """
    POSET priority levels — smaller integer = higher importance.

    Mathematical Structure
    ----------------------
    Models the totally-ordered POSET  (S, ≤)  where the carrier set S is
    partitioned into priority tiers.  The covering relation a ⋖ b holds iff
    b is the immediate successor of a in the Hasse diagram:

        CRITICAL ⋖ HIGH ⋖ MEDIUM ⋖ LOW ⋖ OFFLINE

    Properties of the partial order:
      • Reflexivity   : a ≤ a  for all a ∈ S
      • Antisymmetry  : a ≤ b ∧ b ≤ a  ⟹  a = b
      • Transitivity  : a ≤ b ∧ b ≤ c  ⟹  a ≤ c

    The meet (⊓) of two nodes = node with lower integer (higher priority).
    The join (⊔) of two nodes = node with higher integer (lower priority).
    The lattice top element = OFFLINE (5); bottom element = CRITICAL (1).
    """
    CRITICAL = 1   # Hospitals, Emergency services, Generators
    HIGH     = 2   # Substations, Major hubs
    MEDIUM   = 3   # Commercial zones
    LOW      = 4   # Residential areas
    OFFLINE  = 5   # Decommissioned / failed nodes


# Mapping from NodeType → default PriorityLevel (used during graph init)
_NODE_TYPE_PRIORITY: dict[str, PriorityLevel] = {
    NodeType.GENERATOR:   PriorityLevel.CRITICAL,
    NodeType.HOSPITAL:    PriorityLevel.CRITICAL,
    NodeType.EMERGENCY:   PriorityLevel.CRITICAL,
    NodeType.SUBSTATION:  PriorityLevel.HIGH,
    NodeType.COMMERCIAL:  PriorityLevel.MEDIUM,
    NodeType.RESIDENTIAL: PriorityLevel.LOW,
}

# Edge attribute names (string constants to avoid typos)
EDGE_RESISTANCE   = "resistance"    # Ω  — primary weight for Dijkstra / MST
EDGE_CAPACITY_MW  = "capacity_mw"   # MW — maximum throughput
EDGE_POWER_LOSS   = "power_loss_pct"# %  — percentage power lost on this line
EDGE_VOLTAGE_KV   = "voltage_kv"    # kV — operating voltage
EDGE_STATUS       = "status"        # "active" | "failed" | "degraded"
EDGE_LENGTH_KM    = "length_km"     # km — physical cable length


# ──────────────────────────────────────────────────────────────────────────────
# § 1  DATA-CLASSES (result containers)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CutSetResult:
    """
    Result container for cut-set analysis.

    Attributes
    ----------
    articulation_points : list[int]
        Nodes whose removal disconnects the graph (cut-vertices).
        Detected via Tarjan's DFS bridge-finding algorithm — O(V+E).
    bridges : list[tuple[int, int]]
        Edges whose removal disconnects the graph (cut-edges / bridges).
    connected_components : list[set[int]]
        Each set contains the node IDs of one connected component after
        applying the failed_edges mask.
    blacked_out_zones : dict[PriorityLevel, list[int]]
        Nodes that are isolated, keyed by their priority level.
        Helps dispatchers identify which critical zones are offline.
    is_connected : bool
        True if the (possibly masked) graph is still fully connected.
    num_components : int
        Number of connected components post-failure.
    component_df : pd.DataFrame
        Tabular summary of each component (size, highest priority node inside).
    """
    articulation_points : list[int]          = field(default_factory=list)
    bridges              : list[tuple[int,int]] = field(default_factory=list)
    connected_components : list[set[int]]    = field(default_factory=list)
    blacked_out_zones    : dict[int, list[int]] = field(default_factory=dict)
    is_connected         : bool              = True
    num_components       : int               = 1
    component_df         : pd.DataFrame      = field(default_factory=pd.DataFrame)


@dataclass
class POSETSchedule:
    """
    Result of the POSET / Lattice priority scheduling pass.

    Attributes
    ----------
    restoration_order : list[int]
        Node IDs sorted by priority (CRITICAL first, LOW last).
    priority_groups : dict[int, list[int]]
        Nodes grouped by PriorityLevel integer value.
    power_allocation : dict[int, float]
        Fraction of available generation capacity allocated to each node
        (sums to ≤ 1.0).  Allocation respects the POSET ordering.
    unserved_nodes : list[int]
        Nodes that could not be served due to generation constraints.
    total_available_mw : float
        Total active generator capacity available.
    total_demanded_mw  : float
        Sum of capacity (load) demanded by all non-generator nodes.
    lattice_edges : list[tuple[int, int]]
        Directed edges of the Hasse diagram (covering relations of the POSET).
    """
    restoration_order  : list[int]           = field(default_factory=list)
    priority_groups    : dict[int, list[int]]= field(default_factory=dict)
    power_allocation   : dict[int, float]    = field(default_factory=dict)
    unserved_nodes     : list[int]           = field(default_factory=list)
    total_available_mw : float               = 0.0
    total_demanded_mw  : float               = 0.0
    lattice_edges      : list[tuple[int,int]]= field(default_factory=list)


@dataclass
class DijkstraResult:
    """
    Result of Dijkstra's Single-Source Shortest Path computation.

    Attributes
    ----------
    source : int
        Source node ID (generator / active hub).
    distances : dict[int, float]
        Minimum accumulated resistance (weight) from source to every node.
        math.inf denotes unreachable nodes.
    predecessors : dict[int, int | None]
        Previous node on the shortest path tree.
    shortest_path_to : dict[int, list[int]]
        Full reconstructed path from source → each target node.
    total_nodes_reached : int
        How many nodes are reachable from source.
    """
    source              : int
    distances           : dict[int, float]      = field(default_factory=dict)
    predecessors        : dict[int, int | None] = field(default_factory=dict)
    shortest_path_to    : dict[int, list[int]]  = field(default_factory=dict)
    total_nodes_reached : int                   = 0


@dataclass
class KruskalResult:
    """
    Result of Kruskal's Minimum Spanning Tree computation.

    Attributes
    ----------
    mst_graph : nx.Graph
        The minimum spanning tree as a NetworkX graph (subset of edges).
    mst_edges : list[tuple[int, int, dict]]
        List of (u, v, attr_dict) for each edge in the MST.
    total_weight : float
        Sum of edge resistances in the MST.
    num_components : int
        Number of connected components in the input graph (MST forms a
        spanning *forest* if graph is disconnected).
    is_spanning : bool
        True if MST spans all nodes (graph was connected).
    edge_df : pd.DataFrame
        Tabular view of MST edges with all attributes.
    """
    mst_graph      : nx.Graph                        = field(default_factory=nx.Graph)
    mst_edges      : list[tuple[int,int,dict]]        = field(default_factory=list)
    total_weight   : float                            = 0.0
    num_components : int                              = 1
    is_spanning    : bool                             = True
    edge_df        : pd.DataFrame                     = field(default_factory=pd.DataFrame)


@dataclass
class AuditEvent:
    """
    Historical or real-time event record in the SCADA incident audit log.

    Attributes
    ----------
    timestamp : str
        SCADA wall-clock timestamp (e.g. '12:34:56' or '%Y-%m-%d %H:%M:%S').
    phase : str
        Simulation or recovery phase identifier (e.g. 'FAULT_INJECTION',
        'CUT_SET_ISOLATION', 'POSET_SCHEDULING', 'RECOVERY_COMPLETION').
    event : str
        Detailed description of the action taken and mathematical impact.
    status : str
        Event severity status: 'OK', 'WARN', 'ALERT', or 'INFO'.
    duration_ms : float
        Execution runtime latency of the underlying discrete math pass in ms.
    metadata : dict[str, Any]
        Arbitrary extra metadata key-values (e.g. affected nodes, bridges, mw).
    """
    timestamp   : str
    phase       : str
    event       : str
    status      : str                  = "OK"
    duration_ms : float                = 0.0
    metadata    : dict[str, Any]       = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert audit event to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "phase": self.phase,
            "event": self.event,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
        }


@dataclass
class ScenarioMetrics:
    """
    Quantitative resilience and performance evaluation for a grid simulation state.

    Attributes
    ----------
    scenario_name : str
        Identifier of the scenario or disaster preset.
    timestamp : str
        Simulation timestamp.
    total_power_loss_mw : float
        Aggregate power lost: generation deficit plus transmission line dissipation (MW).
    unserved_demand_mw : float
        Load shedding deficit (demanded MW - served MW).
    transmission_loss_mw : float
        Estimated active line Joule dissipation (MW).
    recovery_latency_ms : float
        End-to-end algorithmic recovery execution latency in ms.
    health_percentage : float
        Composite grid health score [0.0 - 100.0]%.
    restoration_rate : float
        Dynamic grid reachability: percentage of total nodes successfully served with power
        (Active Served Nodes / Total Nodes).
    critical_served_rate : float
        Percentage of critical facilities (Hospitals, Emergency) receiving 100% power.
    active_edges_count : int
        Number of operational transmission lines.
    broken_edges_count : int
        Number of severed transmission lines.
    active_nodes_count : int
        Number of operational substations.
    failed_nodes_count : int
        Number of offline substations.
    num_islands : int
        Number of connected components (islands).
    articulation_points_count : int
        Number of cut-vertices detected via Tarjan's DFS.
    bridges_count : int
        Number of cut-edges (bridges) detected via Tarjan's DFS.
    total_generation_mw : float
        Total active generator capacity dispatch (MW).
    total_demand_mw : float
        Total aggregate network demand (MW).
    mst_resistance_ohm : float
        Total electrical resistance of Kruskal minimum spanning backbone (Ω).
    phase_latencies_ms : dict[str, float]
        Execution time breakdown per recovery phase.
    metadata : dict[str, Any]
        Additional scenario attributes.
    """
    scenario_name             : str
    timestamp                 : str
    total_power_loss_mw       : float
    unserved_demand_mw        : float
    transmission_loss_mw      : float
    recovery_latency_ms       : float
    health_percentage         : float
    restoration_rate          : float
    critical_served_rate      : float
    active_edges_count        : int
    broken_edges_count        : int
    active_nodes_count        : int
    failed_nodes_count        : int
    num_islands               : int
    articulation_points_count : int
    bridges_count             : int
    total_generation_mw       : float
    total_demand_mw           : float
    mst_resistance_ohm        : float
    phase_latencies_ms        : dict[str, float] = field(default_factory=dict)
    metadata                  : dict[str, Any]   = field(default_factory=dict)

    @property
    def grid_reachability(self) -> float:
        """Dynamic grid reachability percentage (Active Served Nodes / Total Nodes)."""
        return self.restoration_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "scenario_name": self.scenario_name,
            "timestamp": self.timestamp,
            "total_power_loss_mw": self.total_power_loss_mw,
            "unserved_demand_mw": self.unserved_demand_mw,
            "transmission_loss_mw": self.transmission_loss_mw,
            "recovery_latency_ms": self.recovery_latency_ms,
            "health_percentage": self.health_percentage,
            "restoration_rate": self.restoration_rate,
            "grid_reachability": self.restoration_rate,
            "critical_served_rate": self.critical_served_rate,
            "active_edges_count": self.active_edges_count,
            "broken_edges_count": self.broken_edges_count,
            "active_nodes_count": self.active_nodes_count,
            "failed_nodes_count": self.failed_nodes_count,
            "num_islands": self.num_islands,
            "articulation_points_count": self.articulation_points_count,
            "bridges_count": self.bridges_count,
            "total_generation_mw": self.total_generation_mw,
            "total_demand_mw": self.total_demand_mw,
            "mst_resistance_ohm": self.mst_resistance_ohm,
            "phase_latencies_ms": self.phase_latencies_ms,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# § 2  GRAPH INITIALISATION  —  G = (V, E, w)
# ──────────────────────────────────────────────────────────────────────────────

def initialize_grid_graph(
    num_nodes: int,
    seed: int = 42,
    connectivity: float = 0.35,
    min_generators: int = 2,
) -> nx.Graph:
    """
    Build a weighted power-distribution network graph G = (V, E, w).

    The graph models a regional smart-grid topology:
      • Nodes → substations, generators, hospitals, commercial & residential zones
      • Edges → high-voltage transmission lines with resistance/loss attributes
      • Edge weight = resistance (Ω), used directly by Dijkstra and Kruskal

    Node Attributes
    ---------------
    node_id        : int             — unique integer identifier
    node_type      : str             — one of NodeType constants
    priority_level : PriorityLevel   — POSET priority (IntEnum 1–5)
    capacity       : float           — MW capacity (generation or load)
    label          : str             — human-readable display name
    status         : str             — "active" | "failed"
    x, y           : float           — 2-D layout coordinates (for visualisation)

    Edge Attributes
    ---------------
    resistance     : float           — Ω,  primary pathfinding weight
    capacity_mw    : float           — MW throughput limit
    power_loss_pct : float           — % power dissipated on this line
    voltage_kv     : float           — kV operating voltage (11/33/66/132 kV)
    length_km      : float           — physical cable length
    status         : str             — "active" | "failed" | "degraded"

    Parameters
    ----------
    num_nodes    : Number of nodes in the grid (recommended 8–50).
    seed         : Random seed for reproducibility.
    connectivity : Probability an edge exists between any pair (Erdős–Rényi p).
    min_generators: Minimum number of Generator nodes to guarantee.

    Returns
    -------
    nx.Graph
        A fully attributed NetworkX undirected weighted graph.

    Examples
    --------
    >>> G = initialize_grid_graph(16)
    >>> G.number_of_nodes()
    16
    >>> 'node_type' in G.nodes[0]
    True
    """
    if num_nodes < 4:
        raise ValueError("num_nodes must be ≥ 4 to form a meaningful grid.")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    G = nx.Graph()

    # ── Determine node-type distribution ────────────────────────────────────
    # Ensure at least min_generators generators; distribute rest probabilistically
    type_pool: list[str] = []
    type_pool.extend([NodeType.GENERATOR] * min_generators)

    remaining = num_nodes - min_generators
    # Weighted distribution for remaining slots
    type_weights = {
        NodeType.HOSPITAL:    0.10,
        NodeType.EMERGENCY:   0.05,
        NodeType.SUBSTATION:  0.25,
        NodeType.COMMERCIAL:  0.25,
        NodeType.RESIDENTIAL: 0.35,
    }
    remaining_types = rng.choices(
        list(type_weights.keys()),
        weights=list(type_weights.values()),
        k=remaining,
    )
    type_pool.extend(remaining_types)
    rng.shuffle(type_pool)

    # Counters for label numbering
    type_counters: dict[str, int] = defaultdict(int)

    # ── Add nodes ────────────────────────────────────────────────────────────
    # Place nodes on a rough circular + jitter layout
    angles = np.linspace(0, 2 * math.pi, num_nodes, endpoint=False)
    radius = 5.0

    for node_id in range(num_nodes):
        ntype = type_pool[node_id]
        type_counters[ntype] += 1
        priority = _NODE_TYPE_PRIORITY[ntype]

        # Capacity: generators produce, others consume
        if ntype == NodeType.GENERATOR:
            capacity = round(float(np_rng.uniform(80.0, 250.0)), 2)   # MW produced
        elif ntype == NodeType.HOSPITAL:
            capacity = round(float(np_rng.uniform(5.0, 20.0)), 2)     # MW consumed
        elif ntype == NodeType.EMERGENCY:
            capacity = round(float(np_rng.uniform(2.0, 8.0)), 2)
        elif ntype == NodeType.SUBSTATION:
            capacity = round(float(np_rng.uniform(30.0, 100.0)), 2)
        elif ntype == NodeType.COMMERCIAL:
            capacity = round(float(np_rng.uniform(10.0, 40.0)), 2)
        else:  # RESIDENTIAL
            capacity = round(float(np_rng.uniform(1.0, 15.0)), 2)

        # 2-D layout with jitter for visualisation
        angle = angles[node_id]
        jitter_r = float(np_rng.uniform(0.3, 1.5))
        x = round((radius + jitter_r) * math.cos(angle), 4)
        y = round((radius + jitter_r) * math.sin(angle), 4)

        label = f"{ntype[0]}{type_counters[ntype]}"  # e.g. G1, H2, S3

        G.add_node(
            node_id,
            node_id       = node_id,
            node_type     = ntype,
            priority_level= int(priority),
            capacity      = capacity,
            label         = label,
            status        = "active",
            x             = x,
            y             = y,
        )

    # ── Add edges (Erdős–Rényi skeleton + guaranteed connectivity) ───────────
    # Step 1: Random edges with probability = connectivity
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng.random() < connectivity:
                G.add_edge(u, v, **_generate_edge_attrs(u, v, np_rng))

    # Step 2: Guarantee full connectivity via random spanning tree
    #         Walk nodes in random order; connect each to a visited predecessor
    nodes_shuffled = list(range(num_nodes))
    rng.shuffle(nodes_shuffled)
    for i in range(1, len(nodes_shuffled)):
        u = nodes_shuffled[i]
        v = nodes_shuffled[rng.randint(0, i - 1)]
        if not G.has_edge(u, v):
            G.add_edge(u, v, **_generate_edge_attrs(u, v, np_rng))

    # Tag the graph with metadata
    G.graph["name"]        = "Smart Power Distribution Grid"
    G.graph["num_nodes"]   = num_nodes
    G.graph["seed"]        = seed
    G.graph["weight_attr"] = EDGE_RESISTANCE

    return G


def _generate_edge_attrs(
    u: int,
    v: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Generate realistic transmission-line attributes for edge (u, v).

    Resistance is modelled as  R = ρ · L / A  (simplified):
      higher voltage lines → lower resistance per km.
    Power loss % is correlated with resistance and length.
    """
    voltage_kv    = float(rng.choice([11.0, 33.0, 66.0, 132.0]))
    length_km     = round(float(rng.uniform(0.5, 50.0)), 2)

    # Resistance: inversely proportional to voltage level (simplified model)
    base_resistance = (1000.0 / voltage_kv) * length_km
    resistance = round(float(rng.uniform(0.8, 1.2)) * base_resistance, 4)

    # Capacity: higher voltage → higher throughput
    capacity_mw = round(float(rng.uniform(voltage_kv * 0.5, voltage_kv * 2.0)), 2)

    # Power loss: 0.5%–8% depending on resistance and length
    # Guard: ensure high > low for short cables (length_km < 3.3 → 0.15*L < 0.5)
    _pct_high      = max(1.0, 0.15 * length_km)
    power_loss_pct = round(min(8.0, float(rng.uniform(0.5, _pct_high))), 3)

    return {
        EDGE_RESISTANCE:  resistance,
        EDGE_CAPACITY_MW: capacity_mw,
        EDGE_POWER_LOSS:  power_loss_pct,
        EDGE_VOLTAGE_KV:  voltage_kv,
        EDGE_LENGTH_KM:   length_km,
        EDGE_STATUS:      "active",
    }


# ──────────────────────────────────────────────────────────────────────────────
# § 3  CUT-SET ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def detect_cut_sets(
    graph: nx.Graph,
    failed_edges: list[tuple[int, int]] | None = None,
    failed_nodes: list[int] | None = None,
) -> CutSetResult:
    """
    Analyse grid connectivity and topological cut-sets after applying a failure mask.

    Mathematical Formulation
    ------------------------
    Let G = (V, E, w) be the unweighted/weighted representation of the power grid.
    Let F_E ⊆ E denote severed transmission lines and F_V ⊆ V denote offline substations.
    The residual operating graph is defined as:
        G' = (V \\ F_V, E \\ (F_E ∪ Inc(F_V)))

    1. Cut-Sets & Bridges (Cut-Edges):
       A cut C = (S, V' \\ S) partitions the residual vertex set.
       The cut-set is the coboundary:
           ∂(S) = { (u, v) ∈ E' : u ∈ S, v ∈ V' \\ S }
       An edge e = (u, v) is a bridge (fundamental cut-set of size 1) iff:
           c(G' \\ {e}) > c(G')
       where c(H) denotes the number of connected components of graph H.
       In Tarjan's DFS traversal with discovery time disc[u] and low-link value low[u]:
           e = (u, v) is a bridge  ⟺  low[v] > disc[u]

    2. Articulation Points (Cut-Vertices):
       A vertex v ∈ V' is an articulation point iff:
           c(G' \\ {v}) > c(G')
       Conditions in DFS tree:
         • Root vertex r is an AP  ⟺  deg_{T_DFS}(r) ≥ 2
         • Non-root vertex u is an AP  ⟺  ∃ child v in T_DFS with low[v] ≥ disc[u]

    3. Blacked-Out Priority Zones:
       Let G' be partitioned into connected components {K_1, K_2, ..., K_m}.
       Component K_i is classified as 'Powered' iff K_i ∩ V_gen ≠ ∅.
       Component K_i is classified as 'Blacked-Out' (Islanding Fault) iff:
           K_i ∩ V_gen = ∅
       All vertices v ∈ K_i are mapped by priority tier to dispatch emergency power.

    Algorithm Complexity
    --------------------
    • Tarjan's Single-Pass DFS (APs & Bridges) : O(|V'| + |E'|) time, O(|V'|) stack space.
    • Connected Component Enumeration (BFS)     : O(|V'| + |E'|) time.
    • Priority Zone Partitioning & Mapping     : O(|V'|) time.
    • Total Run-Time Complexity                : O(|V| + |E|)  — guaranteed < 200 ms for V ≤ 50.

    Parameters
    ----------
    graph        : The baseline grid graph G (preserved non-destructively).
    failed_edges : List of (u, v) tuples for severed transmission lines.
    failed_nodes : List of node IDs for offline/damaged substations.

    Returns
    -------
    CutSetResult
        Container holding articulation points, bridges, connected components,
        blacked-out priority zones, and a summary pandas DataFrame.

    Examples
    --------
    >>> G = initialize_grid_graph(12)
    >>> result = detect_cut_sets(G, failed_edges=[(0, 1), (2, 3)])
    >>> result.num_components >= 1
    True
    """
    failed_edges = [tuple(sorted(e)) for e in (failed_edges or [])]
    failed_nodes = list(failed_nodes or [])

    # ── Build residual graph G' ──────────────────────────────────────────────
    G_prime: nx.Graph = graph.copy()

    # Remove failed nodes (and their incident edges)
    for n in failed_nodes:
        if G_prime.has_node(n):
            G_prime.remove_node(n)
            # Mark as offline in result (not in G_prime)

    # Remove failed edges
    for u, v in failed_edges:
        if G_prime.has_edge(u, v):
            G_prime[u][v][EDGE_STATUS] = "failed"
            G_prime.remove_edge(u, v)

    result = CutSetResult()

    # ── Tarjan's algorithm for articulation points & bridges ─────────────────
    result.articulation_points, result.bridges = _tarjan_cut_analysis(G_prime)

    # ── Connected components ─────────────────────────────────────────────────
    components: list[set[int]] = [
        set(c) for c in nx.connected_components(G_prime)
    ]
    result.connected_components = components
    result.num_components       = len(components)
    result.is_connected         = (result.num_components == 1)

    # ── Identify generator-containing components (powered components) ────────
    powered_components: set[int] = set()   # indices into components list
    for idx, comp in enumerate(components):
        for node in comp:
            if graph.nodes[node].get("node_type") == NodeType.GENERATOR:
                powered_components.add(idx)
                break

    # ── Blacked-out zones: components with NO generator ─────────────────────
    blacked: dict[int, list[int]] = defaultdict(list)
    for idx, comp in enumerate(components):
        if idx not in powered_components:
            for node in comp:
                prio = graph.nodes[node].get("priority_level", int(PriorityLevel.LOW))
                blacked[prio].append(node)

    # Also mark explicitly failed nodes as blacked out
    for node in failed_nodes:
        prio = graph.nodes[node].get("priority_level", int(PriorityLevel.LOW))
        blacked[prio].append(node)

    result.blacked_out_zones = dict(blacked)

    # ── Build component summary DataFrame ───────────────────────────────────
    rows = []
    for idx, comp in enumerate(components):
        has_generator   = idx in powered_components
        node_types_here = [graph.nodes[n].get("node_type", "Unknown") for n in comp]
        priorities_here = [graph.nodes[n].get("priority_level", 5) for n in comp]
        min_priority    = min(priorities_here) if priorities_here else 5

        rows.append({
            "component_id"       : idx,
            "size"               : len(comp),
            "has_generator"      : has_generator,
            "highest_priority"   : min_priority,
            "node_types_present" : ", ".join(sorted(set(node_types_here))),
            "nodes"              : sorted(comp),
        })

    result.component_df = pd.DataFrame(rows)

    return result


def _tarjan_cut_analysis(
    G: nx.Graph,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Tarjan's iterative DFS to find articulation points and bridges in O(V+E).

    Uses an explicit stack to avoid Python's recursion limit on large grids.

    Returns
    -------
    (articulation_points, bridges)
    """
    if G.number_of_nodes() == 0:
        return [], []

    visited   : dict[int, bool]       = {}
    disc      : dict[int, int]        = {}   # Discovery time
    low       : dict[int, int]        = {}   # Lowest disc reachable via back-edges
    parent    : dict[int, int | None] = {}
    ap_set    : set[int]              = set()
    bridges   : list[tuple[int, int]] = []
    timer     : list[int]             = [0]  # mutable counter via list trick

    def dfs(start: int) -> None:
        """Iterative Tarjan DFS from a given start node."""
        # Stack entries: (node, iterator_over_neighbours, child_count)
        stack: list[tuple[int, Generator, int]] = []
        disc[start]    = low[start] = timer[0]
        timer[0]      += 1
        visited[start] = True
        parent[start]  = None
        stack.append((start, iter(G.neighbors(start)), 0))

        while stack:
            u, neighbours, child_count = stack[-1]

            try:
                v = next(neighbours)
            except StopIteration:
                stack.pop()
                if stack:
                    p, _, _ = stack[-1]
                    # Propagate low values upward
                    low[p] = min(low[p], low[u])

                    # Articulation-point check for non-root
                    if parent[p] is not None and low[u] >= disc[p]:
                        ap_set.add(p)

                    # Bridge check
                    if low[u] > disc[p]:
                        bridges.append((p, u))
                continue

            if not visited.get(v, False):
                # Tree edge: go deeper
                disc[v]    = low[v] = timer[0]
                timer[0]  += 1
                visited[v] = True
                parent[v]  = u

                # Update child count on current frame
                stack[-1] = (u, neighbours, child_count + 1)
                stack.append((v, iter(G.neighbors(v)), 0))

            elif v != parent[u]:
                # Back edge: update low
                low[u] = min(low[u], disc[v])
                stack[-1] = (u, neighbours, child_count)
            else:
                stack[-1] = (u, neighbours, child_count)

        # Root articulation-point check (root is AP if it has > 1 DFS children)
        if stack == []:
            # Already popped; check root separately
            pass

    # Run DFS for each unvisited node (handles disconnected graphs)
    all_nodes = list(G.nodes())
    root_children: dict[int, int] = {}

    for start in all_nodes:
        if not visited.get(start, False):
            # Count root's children in DFS tree for AP check
            disc[start]    = low[start] = timer[0]
            timer[0]      += 1
            visited[start] = True
            parent[start]  = None
            child_count_root = 0

            dfs_stack: list[tuple[int, Any, list[int]]] = [
                (start, iter(G.neighbors(start)), [0])
            ]

            while dfs_stack:
                u, nbrs, cc = dfs_stack[-1]
                try:
                    v = next(nbrs)
                except StopIteration:
                    dfs_stack.pop()
                    if dfs_stack:
                        p, _, _ = dfs_stack[-1]
                        low[p] = min(low[p], low[u])
                        if parent[p] is None:   # p is root
                            pass                # root check done below
                        else:
                            if low[u] >= disc[p]:
                                ap_set.add(p)
                        if low[u] > disc[p]:
                            bridges.append(
                                (min(p, u), max(p, u))
                            )
                    continue

                if not visited.get(v, False):
                    disc[v]    = low[v] = timer[0]
                    timer[0]  += 1
                    visited[v] = True
                    parent[v]  = u
                    if u == start:
                        cc[0] += 1
                    dfs_stack.append((v, iter(G.neighbors(v)), [0]))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

            # Root is AP iff it has more than one child in DFS tree
            if cc[0] > 1:
                ap_set.add(start)

    # De-duplicate bridges (undirected graph may produce (u,v) and (v,u))
    unique_bridges = list({(min(u, v), max(u, v)) for u, v in bridges})

    return sorted(ap_set), sorted(unique_bridges)


# ──────────────────────────────────────────────────────────────────────────────
# § 4  POSET — LATTICE PRIORITY SCHEDULER
# ──────────────────────────────────────────────────────────────────────────────

def compute_poset_schedule(
    graph: nx.Graph,
    available_generation_mw: float | None = None,
    failed_nodes: list[int] | None = None,
) -> POSETSchedule:
    """
    Partial-Order Set (POSET) & Lattice Priority Power Restoration Scheduler.

    Mathematical Formulation
    ------------------------
    Let P = V_active be the carrier set of non-failed network entities.
    We endow P with a binary relation ⊑ defining a Partially Ordered Set (P, ⊑):
        u ⊑ v  ⟺  priority_level(u) ≤ priority_level(v)

    1. Mathematical Axioms Satisfied:
       • Reflexivity  : ∀ u ∈ P, u ⊑ u
       • Antisymmetry : ∀ u, v ∈ P, (u ⊑ v ∧ v ⊑ u) ⟹ u ~ v  (quotient order on priority tiers)
       • Transitivity : ∀ u, v, w ∈ P, (u ⊑ v ∧ v ⊑ w) ⟹ u ⊑ w

    2. Total Order Lattice Hierarchy:
       The priority quotient lattice forms a total chain:
           CRITICAL (1) ⊑ HIGH (2) ⊑ MEDIUM (3) ⊑ LOW (4) ⊑ OFFLINE (5)
       For any two tiers a, b ∈ L:
         • Meet (Infimum / Greatest Lower Bound): a ⊓ b = min(a, b)  (higher urgency)
         • Join (Supremum / Least Upper Bound)  : a ⊔ b = max(a, b)  (lower urgency)
       Bottom element ⊥ = CRITICAL (Hospitals / Generators), Top element ⊤ = OFFLINE.

    3. Hasse Diagram Covering Relations:
       A pair (u, v) is an edge in the Hasse Diagram iff v covers u (u ⋖ v):
           u ⋖ v  ⟺  (u ⊏ v) ∧ (∄ z ∈ P such that u ⊏ z ⊏ v)
       Returned in `lattice_edges` for academic and visualization rendering.

    4. Resource Allocation Optimization:
       Let G_avail = total active generation capacity (MW).
       For each non-generator node i ∈ P with demand d_i:
           Allocate x_i ∈ [0, 1] according to the greedy topological sequence:
               x_i = min(1.0, max(0.0, remaining_generation / d_i))
       Theorem (Zero Priority Inversion):
           ∀ u, v ∈ P with priority(u) < priority(v):
               x_v > 0  ⟹  x_u = 1.0
           Hospitals & emergency facilities are guaranteed 100% service before
           any residential zone receives power.

    Algorithm Complexity
    --------------------
    • Antichain Partitioning & Node Sorting : O(|V| log |V|)
    • Greedy Knapsack Power Allocation      : O(|V|)
    • Hasse Covering Relation Construction  : O(|V|)
    • Total Time Complexity                 : O(|V| log |V|)  — runs in < 5 ms for V ≤ 50.

    Parameters
    ----------
    graph                   : The live grid graph G.
    available_generation_mw : Override for total generation capacity (MW).
                              If None, summed from active Generator nodes.
    failed_nodes            : Nodes to exclude from scheduling (disaster mask).

    Returns
    -------
    POSETSchedule
        Container holding restoration_order, priority_groups, power_allocation,
        unserved_nodes, total_available_mw, total_demanded_mw, and lattice_edges.

    Examples
    --------
    >>> G = initialize_grid_graph(16)
    >>> schedule = compute_poset_schedule(G)
    >>> schedule.restoration_order[0] in find_critical_nodes(G)
    True
    """
    failed_nodes = set(failed_nodes or [])
    result       = POSETSchedule()

    # ── Gather active (non-failed) nodes ─────────────────────────────────────
    active_nodes = [
        n for n in graph.nodes()
        if n not in failed_nodes
        and graph.nodes[n].get("status", "active") != "failed"
    ]

    # ── Compute total available generation ───────────────────────────────────
    if available_generation_mw is not None:
        total_gen_mw = float(available_generation_mw)
    else:
        total_gen_mw = sum(
            graph.nodes[n]["capacity"]
            for n in active_nodes
            if graph.nodes[n].get("node_type") == NodeType.GENERATOR
        )

    result.total_available_mw = round(total_gen_mw, 2)

    # ── Compute total demand ─────────────────────────────────────────────────
    total_demand = sum(
        graph.nodes[n]["capacity"]
        for n in active_nodes
        if graph.nodes[n].get("node_type") != NodeType.GENERATOR
    )
    result.total_demanded_mw = round(total_demand, 2)

    # ── Group nodes by priority level (POSET partition into antichains) ───────
    groups: dict[int, list[int]] = defaultdict(list)
    for n in active_nodes:
        prio = graph.nodes[n].get("priority_level", int(PriorityLevel.LOW))
        groups[prio].append(n)

    result.priority_groups = dict(groups)

    # ── Build restoration order (topological sort of the total order) ─────────
    # Since PriorityLevel is a total order, topological sort = sort by prio int
    restoration_order: list[int] = []
    for prio in sorted(groups.keys()):
        # Within same priority, sort generators first, then by capacity descending
        tier_nodes = sorted(
            groups[prio],
            key=lambda n: (
                0 if graph.nodes[n].get("node_type") == NodeType.GENERATOR else 1,
                -graph.nodes[n].get("capacity", 0.0),
            ),
        )
        restoration_order.extend(tier_nodes)

    result.restoration_order = restoration_order

    # ── Power allocation (greedy, respect POSET order) ────────────────────────
    remaining_mw = total_gen_mw
    allocation   : dict[int, float] = {}

    for n in restoration_order:
        ntype    = graph.nodes[n].get("node_type")
        demand   = graph.nodes[n].get("capacity", 0.0)

        if ntype == NodeType.GENERATOR:
            # Generators produce, don't consume allocation
            allocation[n] = 0.0
            continue

        if remaining_mw <= 0.0:
            allocation[n] = 0.0
            result.unserved_nodes.append(n)
        elif demand <= remaining_mw:
            allocation[n]  = round(demand / total_gen_mw, 6) if total_gen_mw > 0 else 0.0
            remaining_mw  -= demand
        else:
            # Partial allocation (degraded service)
            allocation[n]  = round(remaining_mw / total_gen_mw, 6) if total_gen_mw > 0 else 0.0
            remaining_mw   = 0.0

    result.power_allocation = allocation

    # ── Build Hasse diagram covering relations ────────────────────────────────
    # In a total order POSET, the covering relation a ⋖ b means
    # b is the immediate successor of a.
    sorted_priority_levels = sorted(groups.keys())
    lattice_edges: list[tuple[int, int]] = []

    # Node-level Hasse: for adjacent priority tiers, connect representative nodes
    for i in range(len(sorted_priority_levels) - 1):
        lower_tier  = sorted_priority_levels[i]
        higher_tier = sorted_priority_levels[i + 1]
        # Add one representative edge per tier pair (full cross-product optional)
        for u in groups[lower_tier][:3]:    # limit to 3 for diagram clarity
            for v in groups[higher_tier][:3]:
                lattice_edges.append((u, v))

    result.lattice_edges = lattice_edges

    return result


# ── POSET Scheduler Aliases & Entry Points ────────────────────────────────────
POSET_Priority_Scheduler = compute_poset_schedule
poset_priority_scheduler = compute_poset_schedule


# ──────────────────────────────────────────────────────────────────────────────
# § 5  DIJKSTRA'S SHORTEST PATH
# ──────────────────────────────────────────────────────────────────────────────

def dijkstra_shortest_path(
    graph: nx.Graph,
    source: int,
    weight_attr: str = EDGE_RESISTANCE,
    targets: list[int] | None = None,
    exclude_edges: list[tuple[int, int]] | None = None,
    exclude_nodes: list[int] | None = None,
) -> DijkstraResult:
    """
    Dijkstra's Single-Source Shortest Path (SSSP) Minimum-Resistance Rerouting Engine.

    Mathematical Formulation
    ------------------------
    Given weighted graph G = (V, E, w) where w: E → ℝ⁺ models transmission-line
    ohmic resistance (Ω). For source generator s ∈ V_gen, find the shortest path
    tree T_s ⊆ G that minimizes total resistance to all reachable nodes v ∈ V:

        dist*(s, v) = min_{p ∈ P(s, v)} ∑_{e ∈ p} w(e)

    where P(s, v) is the set of all elementary paths connecting s to v.

    1. Mathematical Invariants & Optimality Conditions:
       • Optimal Substructure: If path p = (s, ..., u, v) is optimal, then the subpath
         p' = (s, ..., u) is an optimal path from s to u.
       • Triangle Inequality: ∀ (u, v) ∈ E:
             dist[v] ≤ dist[u] + w(u, v)
       • Greedy Choice Property: At step k, settling vertex u* = argmin_{u ∉ S} dist[u]
         guarantees dist[u*] = dist*(s, u*), provided all edge weights w(e) ≥ 0.

    2. Electrical Grid Application:
       By minimizing accumulated line resistance R_total, the rerouting solver minimizes
       Joule thermal dissipation losses:
           P_loss = I² · R_total
       This ensures emergency power from active generators reaches priority hubs (Hospitals,
       Emergency Services) with maximum delivered voltage stability and minimum attenuation.

    3. Algorithmic Invariants & Implementation:
       • Binary Min-Heap Priority Queue: Maintains frontier of unsettled candidate paths.
       • Lazy Deletion Strategy: Avoids expensive O(|V|) decrease-key operations by ignoring
         stale heap entries when u ∈ settled.
       • Predecessor Pointers: Reconstructs exact minimum-resistance paths in O(path_length).
       • Failure Masking: Explicitly bypasses severed lines and disabled substations.

    Algorithm Complexity
    --------------------
    • Time Complexity  : O((|V| + |E|) log |V|) using binary heap (heapq).
    • Space Complexity : O(|V| + |E|) for distance array, predecessor map, and heap.
    • Practical SLA    : Executed in < 15 ms on 50-node topologies (< 200 ms SLA ceiling).

    Parameters
    ----------
    graph         : Weighted grid graph G.
    source        : Source substation ID (typically a Generator node).
    weight_attr   : Edge weight attribute (default: EDGE_RESISTANCE = 'resistance').
    targets       : Optional target nodes for early-stopping optimization.
    exclude_edges : List of severed lines to ignore (failed transmission edges).
    exclude_nodes : List of disabled substations to ignore (failed nodes).

    Returns
    -------
    DijkstraResult
        Container holding distances (Ω), predecessors, reconstructed shortest paths,
        and total nodes reached.

    Raises
    ------
    ValueError
        If source node is missing from graph or any edge has negative resistance.

    Examples
    --------
    >>> G = initialize_grid_graph(10)
    >>> result = dijkstra_shortest_path(G, source=0)
    >>> result.distances[0] == 0.0
    True
    """
    if source not in graph:
        raise ValueError(f"Source node {source} not found in graph.")

    exclude_edges = {tuple(sorted(e)) for e in (exclude_edges or [])}
    exclude_nodes = set(exclude_nodes or [])
    targets_set   = set(targets) if targets else None

    # Initialise distance and predecessor maps
    dist: dict[int, float]      = {n: math.inf for n in graph.nodes()}
    pred: dict[int, int | None] = {n: None      for n in graph.nodes()}
    dist[source] = 0.0

    # Min-heap: (accumulated_cost, node_id)
    heap: list[tuple[float, int]] = [(0.0, source)]
    settled: set[int] = set()

    while heap:
        current_dist, u = heapq.heappop(heap)

        if u in settled:
            continue                     # Lazy deletion of stale entries
        if u in exclude_nodes:
            continue

        settled.add(u)

        # Early termination once all requested targets are settled
        if targets_set is not None and targets_set.issubset(settled):
            break

        for v in graph.neighbors(u):
            if v in exclude_nodes:
                continue
            edge_key = tuple(sorted((u, v)))
            if edge_key in exclude_edges:
                continue

            edge_data = graph[u][v]
            if edge_data.get(EDGE_STATUS) == "failed":
                continue

            w = edge_data.get(weight_attr, 1.0)
            if w < 0:
                raise ValueError(
                    f"Negative weight on edge ({u}, {v}): {w}. "
                    "Dijkstra requires non-negative weights."
                )

            new_dist = current_dist + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                pred[v] = u
                heapq.heappush(heap, (new_dist, v))

    # ── Reconstruct full paths ────────────────────────────────────────────────
    path_targets = list(targets_set) if targets_set else [
        n for n in graph.nodes() if n != source and dist[n] < math.inf
    ]

    paths: dict[int, list[int]] = {}
    for t in path_targets:
        if dist[t] == math.inf:
            paths[t] = []          # unreachable
        else:
            paths[t] = _reconstruct_path(pred, source, t)

    reached = sum(1 for d in dist.values() if d < math.inf)

    return DijkstraResult(
        source              = source,
        distances           = {k: round(v, 6) for k, v in dist.items()},
        predecessors        = pred,
        shortest_path_to    = paths,
        total_nodes_reached = reached,
    )


def _reconstruct_path(
    pred: dict[int, int | None],
    source: int,
    target: int,
) -> list[int]:
    """Walk predecessor map backwards to reconstruct source→target path."""
    path: list[int] = []
    current: int | None = target
    while current is not None:
        path.append(current)
        current = pred.get(current)
        if current == source:
            path.append(source)
            break
    path.reverse()
    # Validate path starts at source (guards against cycles or disconnection)
    if not path or path[0] != source:
        return []
    return path


# ── Dijkstra Algorithm Aliases ────────────────────────────────────────────────
dijkstra = dijkstra_shortest_path
Dijkstra = dijkstra_shortest_path


# ──────────────────────────────────────────────────────────────────────────────
# § 6  KRUSKAL'S MINIMUM SPANNING TREE
# ──────────────────────────────────────────────────────────────────────────────

def kruskal_mst(
    graph: nx.Graph,
    weight_attr: str  = EDGE_RESISTANCE,
    exclude_edges: list[tuple[int, int]] | None = None,
    exclude_nodes: list[int] | None = None,
) -> KruskalResult:
    """
    Kruskal's Minimum Spanning Tree (MST) & Minimum Spanning Forest (MSF) Engine.

    Mathematical Formulation
    ------------------------
    Given a weighted undirected power grid G = (V, E, w) where w: E → ℝ⁺ models
    line resistance (Ω). For surviving active vertices V_active ⊆ V and surviving
    edges E_active ⊆ E, find an acyclic subgraph T = (V_active, E_T) with E_T ⊆ E_active
    that minimizes total electrical impedance:

        w(T) = ∑_{e ∈ E_T} w(e)  ⟶  min!

    subject to:
      1. Spanning Invariant: For each connected component C_k of G_residual, T[C_k]
         spans all vertices in C_k.
      2. Acyclic Invariant: T contains zero cycles (|E_T| = |V_active| - c(G_residual)).

    1. Mathematical Cut Property Theorem:
       Let C = (S, V_active \\ S) be any non-trivial cut in G_residual. If edge
       e* = argmin_{e ∈ ∂(S)} w(e) is a minimum-weight edge crossing the cut boundary ∂(S),
       then e* belongs to an optimal Minimum Spanning Tree T*.
       Kruskal's algorithm greedily operationalizes this theorem by processing edges in
       strictly non-decreasing weight order.

    2. Disjoint Set Union (DSU / Union-Find) Invariants:
       Cycle prevention is managed via an optimized DSU with two structural invariants:
       • Path Compression: Flattens tree pointers during find(x), pointing all ancestors
         directly to the canonical root:
             parent[x] ⟵ root
       • Union by Rank: Always attaches the shallower tree under the deeper tree root:
             rank[r_smaller] < rank[r_larger] ⟹ parent[r_smaller] = r_larger
       Theorem (Tarjan & Hopcroft):
           The sequence of m operations on n elements takes O(m · α(n)) time, where α
           is the inverse Ackermann function (α(n) ≤ 4 for all n < 10^80).

    3. Disaster Recovery Semantics:
       When a disaster partitions the grid into k disconnected islands (c(G_residual) = k > 1),
       Kruskal's algorithm gracefully produces a Minimum Spanning Forest (MSF) with:
           |E_T| = |V_active| - k
       This identifies the cheapest surviving transmission backbone to re-energize each
       island independently without closed-loop circulating currents.

    Algorithm Complexity
    --------------------
    • Edge Sorting Phase       : O(|E| log |E|) time.
    • DSU Cycle Checking Phase : O(|E| · α(|V|)) ≈ O(|E|) time.
    • Total Time Complexity    : O(|E| log |E|) = O(|E| log |V|).
    • Space Complexity         : O(|V| + |E|) for DSU parent/rank tables and edge list.
    • Practical SLA            : Executed in < 10 ms on 50-node topologies (< 200 ms SLA).

    Parameters
    ----------
    graph         : The power grid graph (original baseline or damaged).
    weight_attr   : Edge weight attribute (default: EDGE_RESISTANCE = 'resistance').
    exclude_edges : List of severed transmission lines to ignore.
    exclude_nodes : List of disabled substations to ignore.

    Returns
    -------
    KruskalResult
        Container holding mst_graph (NetworkX), mst_edges, total_weight (Ω),
        num_components, is_spanning flag, and an edge summary pandas DataFrame.

    Examples
    --------
    >>> G = initialize_grid_graph(16)
    >>> result = kruskal_mst(G)
    >>> result.is_spanning
    True
    >>> len(result.mst_edges) == G.number_of_nodes() - 1
    True
    """
    exclude_edges  = {tuple(sorted(e)) for e in (exclude_edges or [])}
    exclude_nodes  = set(exclude_nodes or [])

    # ── Filter active nodes ──────────────────────────────────────────────────
    active_nodes = [
        n for n in graph.nodes()
        if n not in exclude_nodes
        and graph.nodes[n].get("status", "active") != "failed"
    ]
    n_active = len(active_nodes)

    # ── Collect and sort active edges by weight ──────────────────────────────
    sorted_edges: list[tuple[float, int, int, dict]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for u, v, data in graph.edges(data=True):
        if u in exclude_nodes or v in exclude_nodes:
            continue
        key = tuple(sorted((u, v)))
        if key in exclude_edges or key in seen_pairs:
            continue
        if data.get(EDGE_STATUS) == "failed":
            continue

        w = data.get(weight_attr, 1.0)
        sorted_edges.append((w, u, v, data))
        seen_pairs.add(key)

    sorted_edges.sort(key=lambda x: x[0])

    # ── Kruskal with Union-Find ──────────────────────────────────────────────
    uf    = _UnionFind(active_nodes)
    mst_e : list[tuple[int, int, dict]] = []
    total_w = 0.0

    for w, u, v, data in sorted_edges:
        if uf.find(u) != uf.find(v):       # different components → safe to add
            uf.union(u, v)
            mst_e.append((u, v, dict(data)))
            total_w += w

        if len(mst_e) == n_active - 1:     # MST complete (spanning tree has V-1 edges)
            break

    # ── Build MST NetworkX graph ─────────────────────────────────────────────
    mst_graph = nx.Graph()
    for n in active_nodes:
        mst_graph.add_node(n, **graph.nodes[n])
    for u, v, data in mst_e:
        mst_graph.add_edge(u, v, **data)

    # ── Determine spanning-ness ──────────────────────────────────────────────
    num_components = nx.number_connected_components(mst_graph)
    is_spanning    = (num_components == 1) and (mst_graph.number_of_nodes() == n_active)

    # ── Build edge summary DataFrame ─────────────────────────────────────────
    edge_rows = []
    for u, v, data in mst_e:
        edge_rows.append({
            "from"          : u,
            "to"            : v,
            "from_label"    : graph.nodes[u].get("label", str(u)),
            "to_label"      : graph.nodes[v].get("label", str(v)),
            EDGE_RESISTANCE : round(data.get(EDGE_RESISTANCE, 0.0), 4),
            EDGE_CAPACITY_MW: data.get(EDGE_CAPACITY_MW, 0.0),
            EDGE_POWER_LOSS : data.get(EDGE_POWER_LOSS, 0.0),
            EDGE_VOLTAGE_KV : data.get(EDGE_VOLTAGE_KV, 0.0),
            EDGE_LENGTH_KM  : data.get(EDGE_LENGTH_KM, 0.0),
        })

    return KruskalResult(
        mst_graph      = mst_graph,
        mst_edges      = mst_e,
        total_weight   = round(total_w, 4),
        num_components = num_components,
        is_spanning    = is_spanning,
        edge_df        = pd.DataFrame(edge_rows),
    )


class _UnionFind:
    """
    Union-Find (Disjoint Set Union) with path compression and union-by-rank.

    Used internally by Kruskal's MST algorithm to detect cycles in O(α(V))
    amortised time per operation, where α is the inverse Ackermann function
    (effectively constant for all practical graph sizes).

    Data Structures
    ---------------
    parent : dict[int, int]   — parent pointer for each node (root → itself)
    rank   : dict[int, int]   — upper bound on subtree height for union-by-rank

    Operations
    ----------
    find(x)    : O(α(n)) amortised  — path-compressed canonical root of x
    union(x,y) : O(α(n)) amortised  — merges components; smaller rank under larger

    Invariant: After union-by-rank, tree height ≤ ⌊log₂(n)⌋, giving the
    O(log n) worst-case before path compression kicks in.
    """

    def __init__(self, nodes: list[int]) -> None:
        self.parent: dict[int, int] = {n: n for n in nodes}
        self.rank  : dict[int, int] = {n: 0  for n in nodes}

    def find(self, x: int) -> int:
        """Path-compressed find."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression: point all nodes on path directly to root
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        """
        Union by rank. Returns True if a merge happened, False if already same set.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


# ──────────────────────────────────────────────────────────────────────────────
# § 7  COMPOSITE RECOVERY PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_full_recovery_pipeline(
    graph: nx.Graph,
    failed_edges: list[tuple[int, int]] | None = None,
    failed_nodes: list[int] | None = None,
) -> dict[str, Any]:
    """
    End-to-end disaster recovery pipeline combining all DM modules.

    Execution Order
    ---------------
    1. **Cut-Set Analysis**  → identify disconnected zones & black-outs
    2. **POSET Scheduling**  → rank surviving nodes by priority
    3. **Kruskal MST**       → build minimum-cost reconnection backbone
    4. **Dijkstra SSSP**     → compute optimal rerouting from every generator

    Parameters
    ----------
    graph        : Original grid graph G.
    failed_edges : List of failed transmission lines.
    failed_nodes : List of failed substations / nodes.

    Returns
    -------
    dict with keys:
        "cut_set"  : CutSetResult
        "poset"    : POSETSchedule
        "mst"      : KruskalResult
        "dijkstra" : dict[int, DijkstraResult]  (one per active generator)
        "summary"  : dict  (high-level KPIs)
    """
    failed_edges = failed_edges or []
    failed_nodes = failed_nodes or []

    # ── Phase 1: Cut-Set Analysis ────────────────────────────────────────────
    cut_result = detect_cut_sets(graph, failed_edges, failed_nodes)

    # ── Phase 2: POSET Scheduling ────────────────────────────────────────────
    poset_result = compute_poset_schedule(graph, failed_nodes=failed_nodes)

    # ── Phase 3: Kruskal MST on residual graph ───────────────────────────────
    mst_result = kruskal_mst(
        graph,
        exclude_edges=failed_edges,
        exclude_nodes=failed_nodes,
    )

    # ── Phase 4: Dijkstra from every active generator ────────────────────────
    active_generators = [
        n for n in graph.nodes()
        if n not in failed_nodes
        and graph.nodes[n].get("node_type") == NodeType.GENERATOR
        and graph.nodes[n].get("status", "active") != "failed"
    ]

    dijkstra_results: dict[int, DijkstraResult] = {}
    for gen in active_generators:
        dijkstra_results[gen] = dijkstra_shortest_path(
            graph,
            source        = gen,
            exclude_edges = failed_edges,
            exclude_nodes = failed_nodes,
        )

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    critical_blacked = len(
        cut_result.blacked_out_zones.get(int(PriorityLevel.CRITICAL), [])
    )
    summary = {
        "total_nodes"            : graph.number_of_nodes(),
        "total_edges"            : graph.number_of_edges(),
        "failed_edges_count"     : len(failed_edges),
        "failed_nodes_count"     : len(failed_nodes),
        "connected_components"   : cut_result.num_components,
        "grid_is_connected"      : cut_result.is_connected,
        "articulation_points"    : len(cut_result.articulation_points),
        "bridges"                : len(cut_result.bridges),
        "critical_nodes_blacked" : critical_blacked,
        "active_generators"      : len(active_generators),
        "mst_total_resistance"   : mst_result.total_weight,
        "mst_is_spanning"        : mst_result.is_spanning,
        "generation_mw"          : poset_result.total_available_mw,
        "demand_mw"              : poset_result.total_demanded_mw,
        "unserved_nodes_count"   : len(poset_result.unserved_nodes),
        "power_deficit_mw"       : round(
            max(0.0, poset_result.total_demanded_mw - poset_result.total_available_mw), 2
        ),
    }

    return {
        "cut_set"  : cut_result,
        "poset"    : poset_result,
        "mst"      : mst_result,
        "dijkstra" : dijkstra_results,
        "summary"  : summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# § 8  CONVENIENCE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def get_node_dataframe(graph: nx.Graph) -> pd.DataFrame:
    """Return all node attributes as a tidy pandas DataFrame."""
    rows = []
    for n, data in graph.nodes(data=True):
        rows.append({"node_id": n, **data})
    return pd.DataFrame(rows)


def get_edge_dataframe(graph: nx.Graph) -> pd.DataFrame:
    """Return all edge attributes as a tidy pandas DataFrame."""
    rows = []
    for u, v, data in graph.edges(data=True):
        rows.append({
            "from"   : u,
            "to"     : v,
            "from_label": graph.nodes[u].get("label", str(u)),
            "to_label"  : graph.nodes[v].get("label", str(v)),
            **data,
        })
    return pd.DataFrame(rows)


def apply_disaster(
    graph: nx.Graph,
    failed_edge_ids: list[tuple[int, int]] | None = None,
    failed_node_ids: list[int] | None = None,
) -> nx.Graph:
    """
    Return a *copy* of the graph with failed elements marked (non-destructive).

    Implements a non-destructive disaster masking operation:
      G' = G with status attributes overwritten for failed elements.

    Failed edges  → status = 'failed'    (not removed; rendered red/dashed)
    Failed nodes  → status = 'failed', priority_level = OFFLINE (5)

    The original graph G is never mutated, so the UI can simultaneously
    display both the pre-disaster topology and the post-disaster state.

    Parameters
    ----------
    graph          : Original grid graph G (not modified).
    failed_edge_ids: List of (u, v) tuples for failed transmission lines.
    failed_node_ids: List of node IDs for offline substations.

    Returns
    -------
    nx.Graph — A copy G' with failure marks applied.
    """
    G = graph.copy()
    for u, v in (failed_edge_ids or []):
        if G.has_edge(u, v):
            G[u][v][EDGE_STATUS] = "failed"
    for n in (failed_node_ids or []):
        if G.has_node(n):
            G.nodes[n]["status"]         = "failed"
            G.nodes[n]["priority_level"] = int(PriorityLevel.OFFLINE)
    return G


def find_generator_nodes(graph: nx.Graph) -> list[int]:
    """Return node IDs of all Generator-type nodes in the graph."""
    return [
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") == NodeType.GENERATOR
    ]


def find_critical_nodes(graph: nx.Graph) -> list[int]:
    """Return node IDs of all CRITICAL priority nodes (Hospitals, Emergency, Generators)."""
    return [
        n for n, d in graph.nodes(data=True)
        if d.get("priority_level", int(PriorityLevel.LOW)) == int(PriorityLevel.CRITICAL)
    ]


def grid_metrics(graph: nx.Graph) -> dict[str, Any]:
    """
    Compute standard graph-theoretic metrics for the grid.

    Metrics Computed
    ----------------
    density           : |E| / (|V| · (|V|-1) / 2)  — edge density ∈ [0, 1]
    avg_degree        : (2|E|) / |V|                — average node degree
    avg_clustering    : mean local clustering coefficient  C(v)
    is_connected      : True iff the graph has exactly 1 connected component
    num_components    : number of weakly connected components
    avg_resistance_ohm: arithmetic mean of edge resistance weights (Ω)
    total_resistance  : sum of all edge resistance weights (Ω)
    diameter          : longest shortest path in hops (-1 if disconnected)

    Parameters
    ----------
    graph : nx.Graph — The power grid graph.

    Returns
    -------
    dict[str, Any]
        Keys: num_nodes, num_edges, density, avg_degree, avg_clustering,
              is_connected, num_components, avg_resistance_ohm,
              total_resistance, diameter.
    """
    degrees    = [d for _, d in graph.degree()]
    resistances= [
        data.get(EDGE_RESISTANCE, 0.0)
        for _, _, data in graph.edges(data=True)
    ]
    return {
        "num_nodes"         : graph.number_of_nodes(),
        "num_edges"         : graph.number_of_edges(),
        "density"           : round(nx.density(graph), 4),
        "avg_degree"        : round(sum(degrees) / len(degrees), 2) if degrees else 0.0,
        "avg_clustering"    : round(nx.average_clustering(graph), 4),
        "is_connected"      : nx.is_connected(graph),
        "num_components"    : nx.number_connected_components(graph),
        "avg_resistance_ohm": round(sum(resistances) / len(resistances), 4) if resistances else 0.0,
        "total_resistance"  : round(sum(resistances), 4),
        "diameter"          : nx.diameter(graph) if nx.is_connected(graph) else -1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# § 9  INCIDENT AUDIT & SCENARIO EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def create_audit_event(
    phase: str,
    event: str,
    status: str = "OK",
    duration_ms: float = 0.0,
    timestamp: str | None = None,
    **metadata: Any,
) -> AuditEvent:
    """
    Factory helper to instantiate an AuditEvent with current or explicit timestamp.

    Parameters
    ----------
    phase : str
        Simulation or recovery phase identifier (e.g. 'FAULT_INJECTION', 'CUT_SET_ISOLATION').
    event : str
        Detailed description of the action taken and mathematical impact.
    status : str
        Severity status: 'OK', 'WARN', 'ALERT', or 'INFO'.
    duration_ms : float
        Runtime execution latency in ms.
    timestamp : str, optional
        Custom timestamp string; defaults to current system time '%H:%M:%S'.
    **metadata : Any
        Key-value metadata parameters stored in the event record.

    Returns
    -------
    AuditEvent
    """
    import time
    ts = timestamp or time.strftime("%H:%M:%S")
    return AuditEvent(
        timestamp=ts,
        phase=phase,
        event=event,
        status=status,
        duration_ms=round(duration_ms, 2),
        metadata=metadata,
    )


def get_disaster_preset_config(
    graph: nx.Graph,
    preset_name: str,
) -> tuple[list[tuple[int, int]], list[int], float]:
    """
    Translate a named disaster preset into severed edges, disabled nodes, and generation scale.

    Parameters
    ----------
    graph : nx.Graph
        The intact baseline grid graph.
    preset_name : str
        Preset label (e.g. 'Category 5 Cyclone', 'Cascading Islanding',
        'Hospital Medical District Isolation', 'Severe Brownout',
        'Worst-Case Dual-Fault', 'Nominal Baseline').

    Returns
    -------
    tuple of (failed_edges, failed_nodes, generation_scale)
    """
    all_edges = sorted(graph.edges())
    all_nodes = sorted(graph.nodes())

    hosp_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") in (NodeType.HOSPITAL, NodeType.EMERGENCY)
    ]
    hosp_incident = [
        (u, v) for u, v in all_edges if u in hosp_nodes or v in hosp_nodes
    ]

    failed_edges: list[tuple[int, int]] = []
    failed_nodes: list[int] = []
    gen_scale: float = 1.0

    preset_lower = preset_name.lower()
    if "cyclone" in preset_lower:
        failed_edges = all_edges[:min(4, len(all_edges))]
    elif "islanding" in preset_lower or "articulation" in preset_lower:
        intact_aps = list(nx.articulation_points(graph))
        if intact_aps:
            failed_nodes = intact_aps[:2]
        else:
            deg_sorted = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
            failed_nodes = [n for n, _ in deg_sorted[:min(2, len(deg_sorted))]]
        failed_edges = all_edges[:min(2, len(all_edges))]
    elif "hospital" in preset_lower:
        failed_edges = hosp_incident[:min(3, len(hosp_incident))]
    elif "brownout" in preset_lower:
        gen_scale = 0.45
    elif "dual-fault" in preset_lower or "worst-case" in preset_lower:
        failed_edges = all_edges[:min(3, len(all_edges))]
        failed_nodes = all_nodes[:min(1, len(all_nodes))]
        gen_scale = 0.50
    elif "nominal" in preset_lower or "baseline" in preset_lower:
        failed_edges = []
        failed_nodes = []
        gen_scale = 1.0

    return failed_edges, failed_nodes, gen_scale


def evaluate_scenario(
    base_graph: nx.Graph,
    failed_edges: list[tuple[int, int]] | None = None,
    failed_nodes: list[int] | None = None,
    available_generation_mw: float | None = None,
    scenario_name: str = "Scenario",
) -> ScenarioMetrics:
    """
    Execute full quantitative assessment of a disaster preset or simulation state.

    Calculates:
      - Total Power Loss (MW): unserved demand deficit + line dissipation
      - Recovery Latency SLA (ms): across all 4 algorithmic phases
      - Composite Grid Health Score (%): reachability, critical load, and topology connectivity
      - Topological cut-set parameters and Kruskal MST resistance
    """
    import time

    failed_edges = [tuple(sorted(e)) for e in (failed_edges or [])]
    failed_nodes = list(failed_nodes or [])

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    phase_times: dict[str, float] = {}

    # Phase 1: Apply Disaster
    t0 = time.perf_counter()
    damaged = apply_disaster(
        base_graph,
        failed_edge_ids=failed_edges or None,
        failed_node_ids=failed_nodes or None,
    )
    phase_times["impact_sim"] = round((time.perf_counter() - t0) * 1000, 2)

    # Phase 2: Tarjan Cut-Set DFS
    t1 = time.perf_counter()
    cut_res = detect_cut_sets(
        damaged,
        failed_edges=failed_edges or None,
        failed_nodes=failed_nodes or None,
    )
    phase_times["cut_set"] = round((time.perf_counter() - t1) * 1000, 2)

    # Phase 3: POSET Scheduling
    t2 = time.perf_counter()
    poset_res = compute_poset_schedule(
        damaged,
        available_generation_mw=available_generation_mw,
        failed_nodes=failed_nodes or None,
    )
    phase_times["poset"] = round((time.perf_counter() - t2) * 1000, 2)

    # Phase 4: Kruskal MST + Dijkstra SSSP
    t3 = time.perf_counter()
    mst_res = kruskal_mst(
        damaged,
        exclude_edges=failed_edges or None,
        exclude_nodes=failed_nodes or None,
    )
    active_generators = [
        n for n in base_graph.nodes()
        if n not in failed_nodes
        and base_graph.nodes[n].get("node_type") == NodeType.GENERATOR
    ]
    for gen in active_generators:
        dijkstra_shortest_path(
            damaged,
            source=gen,
            exclude_edges=failed_edges or None,
            exclude_nodes=failed_nodes or None,
        )
    phase_times["routing_mst"] = round((time.perf_counter() - t3) * 1000, 2)

    total_latency_ms = round(sum(phase_times.values()), 2)

    # Power metrics
    unserved_mw = round(
        max(0.0, poset_res.total_demanded_mw - poset_res.total_available_mw), 2
    )
    tx_loss_mw = 0.0
    for _, _, d in damaged.edges(data=True):
        if d.get(EDGE_STATUS, "active") == "active":
            cap = float(d.get(EDGE_CAPACITY_MW, 50.0))
            loss_pct = float(d.get(EDGE_POWER_LOSS, 2.5))
            tx_loss_mw += cap * (loss_pct / 100.0) * 0.1
    tx_loss_mw = round(tx_loss_mw, 2)
    total_power_loss_mw = round(unserved_mw + tx_loss_mw, 2)

    # Reachability & Critical Nodes
    reachable: set[int] = set()
    for gn in active_generators:
        if gn in damaged and gn not in failed_nodes:
            reachable |= nx.node_connected_component(damaged, gn)

    live_nodes = [n for n in base_graph.nodes() if n not in failed_nodes]
    total_nodes = base_graph.number_of_nodes()

    # Dynamic Reachability: percentage of nodes that are successfully served with power
    # (Active Served Nodes / Total Nodes)
    if poset_res.total_available_mw > 0.0 and active_generators:
        active_served_nodes = [
            n for n in base_graph.nodes()
            if n in live_nodes and n in reachable and n not in poset_res.unserved_nodes
        ]
    else:
        active_served_nodes = []

    restoration_rate = (
        round(len(active_served_nodes) / total_nodes * 100, 1)
        if total_nodes > 0
        else 0.0
    )

    critical_nodes = [
        n for n in base_graph.nodes()
        if base_graph.nodes[n].get("priority_level", 5) == int(PriorityLevel.CRITICAL)
    ]
    crit_served = (
        [
            n for n in critical_nodes
            if n in live_nodes and n in reachable and n not in poset_res.unserved_nodes
        ]
        if poset_res.total_available_mw > 0.0 and active_generators
        else []
    )
    critical_served_rate = (
        round(len(crit_served) / len(critical_nodes) * 100, 1)
        if critical_nodes
        else 100.0
    )

    # Composite Grid Health %
    active_edges_count = sum(
        1 for _, _, d in damaged.edges(data=True)
        if d.get(EDGE_STATUS, "active") == "active"
    )
    total_edges_count = base_graph.number_of_edges()
    edge_health_rate = (
        (active_edges_count / total_edges_count * 100)
        if total_edges_count > 0
        else 0.0
    )

    comp_factor = (1.0 / max(1, cut_res.num_components)) * 100.0
    power_factor = (
        min(1.0, (poset_res.total_available_mw / max(1.0, poset_res.total_demanded_mw)))
        * 100.0
    )

    health_pct = round(
        0.30 * restoration_rate +
        0.25 * critical_served_rate +
        0.20 * edge_health_rate +
        0.15 * comp_factor +
        0.10 * power_factor,
        1,
    )
    health_pct = min(100.0, max(0.0, health_pct))

    return ScenarioMetrics(
        scenario_name             = scenario_name,
        timestamp                 = ts,
        total_power_loss_mw       = total_power_loss_mw,
        unserved_demand_mw        = unserved_mw,
        transmission_loss_mw      = tx_loss_mw,
        recovery_latency_ms       = total_latency_ms,
        health_percentage         = health_pct,
        restoration_rate          = restoration_rate,
        critical_served_rate      = critical_served_rate,
        active_edges_count        = active_edges_count,
        broken_edges_count        = len(failed_edges),
        active_nodes_count        = len(live_nodes),
        failed_nodes_count        = len(failed_nodes),
        num_islands               = cut_res.num_components,
        articulation_points_count = len(cut_res.articulation_points),
        bridges_count             = len(cut_res.bridges),
        total_generation_mw       = poset_res.total_available_mw,
        total_demand_mw           = poset_res.total_demanded_mw,
        mst_resistance_ohm        = mst_res.total_weight,
        phase_latencies_ms        = phase_times,
        metadata                  = {
            "active_served_nodes": active_served_nodes,
            "active_served_nodes_count": len(active_served_nodes),
            "total_nodes": total_nodes,
        },
    )


def compare_scenario_metrics(
    metrics_a: ScenarioMetrics,
    metrics_b: ScenarioMetrics,
) -> pd.DataFrame:
    """
    Produce a side-by-side comparative analysis DataFrame between two scenarios.

    Computes absolute and percentage differentials:
        Delta = Metric(B) - Metric(A)
    """
    rows = [
        {
            "Category": "Power Loss",
            "Metric": "Total Power Loss (MW)",
            "Scenario A": f"{metrics_a.total_power_loss_mw:.2f}",
            "Scenario B": f"{metrics_b.total_power_loss_mw:.2f}",
            "Delta (B - A)": f"{metrics_b.total_power_loss_mw - metrics_a.total_power_loss_mw:+.2f}",
            "Change (%)": f"{((metrics_b.total_power_loss_mw - metrics_a.total_power_loss_mw) / max(0.01, metrics_a.total_power_loss_mw)) * 100:+.1f}%",
            "Status": "Degraded" if metrics_b.total_power_loss_mw > metrics_a.total_power_loss_mw else ("Improved" if metrics_b.total_power_loss_mw < metrics_a.total_power_loss_mw else "Nominal"),
        },
        {
            "Category": "Power Loss",
            "Metric": "Unserved Demand / Deficit (MW)",
            "Scenario A": f"{metrics_a.unserved_demand_mw:.2f}",
            "Scenario B": f"{metrics_b.unserved_demand_mw:.2f}",
            "Delta (B - A)": f"{metrics_b.unserved_demand_mw - metrics_a.unserved_demand_mw:+.2f}",
            "Change (%)": f"{((metrics_b.unserved_demand_mw - metrics_a.unserved_demand_mw) / max(0.01, metrics_a.unserved_demand_mw)) * 100:+.1f}%" if metrics_a.unserved_demand_mw > 0 else ("+100%" if metrics_b.unserved_demand_mw > 0 else "0.0%"),
            "Status": "Degraded" if metrics_b.unserved_demand_mw > metrics_a.unserved_demand_mw else ("Improved" if metrics_b.unserved_demand_mw < metrics_a.unserved_demand_mw else "Nominal"),
        },
        {
            "Category": "Performance",
            "Metric": "Recovery Latency SLA (ms)",
            "Scenario A": f"{metrics_a.recovery_latency_ms:.2f}",
            "Scenario B": f"{metrics_b.recovery_latency_ms:.2f}",
            "Delta (B - A)": f"{metrics_b.recovery_latency_ms - metrics_a.recovery_latency_ms:+.2f}",
            "Change (%)": f"{((metrics_b.recovery_latency_ms - metrics_a.recovery_latency_ms) / max(0.01, metrics_a.recovery_latency_ms)) * 100:+.1f}%",
            "Status": "Optimal (<200ms)" if metrics_b.recovery_latency_ms < 200 else "Warning (>200ms)",
        },
        {
            "Category": "Grid Resilience",
            "Metric": "Composite Health Score (%)",
            "Scenario A": f"{metrics_a.health_percentage:.1f}%",
            "Scenario B": f"{metrics_b.health_percentage:.1f}%",
            "Delta (B - A)": f"{metrics_b.health_percentage - metrics_a.health_percentage:+.1f}%",
            "Change (%)": f"{(metrics_b.health_percentage - metrics_a.health_percentage):+.1f}%",
            "Status": "Degraded" if metrics_b.health_percentage < metrics_a.health_percentage else ("Improved" if metrics_b.health_percentage > metrics_a.health_percentage else "Nominal"),
        },
        {
            "Category": "Grid Resilience",
            "Metric": "Grid Reachability / Restoration (%)",
            "Scenario A": f"{metrics_a.restoration_rate:.1f}%",
            "Scenario B": f"{metrics_b.restoration_rate:.1f}%",
            "Delta (B - A)": f"{metrics_b.restoration_rate - metrics_a.restoration_rate:+.1f}%",
            "Change (%)": f"{(metrics_b.restoration_rate - metrics_a.restoration_rate):+.1f}%",
            "Status": "Degraded" if metrics_b.restoration_rate < metrics_a.restoration_rate else ("Improved" if metrics_b.restoration_rate > metrics_a.restoration_rate else "Nominal"),
        },
        {
            "Category": "Grid Resilience",
            "Metric": "Critical Facilities Powered (%)",
            "Scenario A": f"{metrics_a.critical_served_rate:.1f}%",
            "Scenario B": f"{metrics_b.critical_served_rate:.1f}%",
            "Delta (B - A)": f"{metrics_b.critical_served_rate - metrics_a.critical_served_rate:+.1f}%",
            "Change (%)": f"{(metrics_b.critical_served_rate - metrics_a.critical_served_rate):+.1f}%",
            "Status": "Degraded" if metrics_b.critical_served_rate < metrics_a.critical_served_rate else "Nominal",
        },
        {
            "Category": "Topology Impact",
            "Metric": "Connected Islands (Components)",
            "Scenario A": f"{metrics_a.num_islands}",
            "Scenario B": f"{metrics_b.num_islands}",
            "Delta (B - A)": f"{metrics_b.num_islands - metrics_a.num_islands:+d}",
            "Change (%)": f"{((metrics_b.num_islands - metrics_a.num_islands) / max(1, metrics_a.num_islands)) * 100:+.1f}%",
            "Status": "Degraded" if metrics_b.num_islands > metrics_a.num_islands else "Nominal",
        },
        {
            "Category": "Topology Impact",
            "Metric": "Severed Transmission Lines",
            "Scenario A": f"{metrics_a.broken_edges_count}",
            "Scenario B": f"{metrics_b.broken_edges_count}",
            "Delta (B - A)": f"{metrics_b.broken_edges_count - metrics_a.broken_edges_count:+d}",
            "Change (%)": "N/A" if metrics_a.broken_edges_count == 0 else f"{((metrics_b.broken_edges_count - metrics_a.broken_edges_count) / metrics_a.broken_edges_count) * 100:+.1f}%",
            "Status": "Degraded" if metrics_b.broken_edges_count > metrics_a.broken_edges_count else "Nominal",
        },
        {
            "Category": "Topology Impact",
            "Metric": "Disabled Substations / Busses",
            "Scenario A": f"{metrics_a.failed_nodes_count}",
            "Scenario B": f"{metrics_b.failed_nodes_count}",
            "Delta (B - A)": f"{metrics_b.failed_nodes_count - metrics_a.failed_nodes_count:+d}",
            "Change (%)": "N/A" if metrics_a.failed_nodes_count == 0 else f"{((metrics_b.failed_nodes_count - metrics_a.failed_nodes_count) / metrics_a.failed_nodes_count) * 100:+.1f}%",
            "Status": "Degraded" if metrics_b.failed_nodes_count > metrics_a.failed_nodes_count else "Nominal",
        },
        {
            "Category": "Cut-Set Analysis",
            "Metric": "Tarjan Bridges (Cut-Edges)",
            "Scenario A": f"{metrics_a.bridges_count}",
            "Scenario B": f"{metrics_b.bridges_count}",
            "Delta (B - A)": f"{metrics_b.bridges_count - metrics_a.bridges_count:+d}",
            "Change (%)": "N/A" if metrics_a.bridges_count == 0 else f"{((metrics_b.bridges_count - metrics_a.bridges_count) / metrics_a.bridges_count) * 100:+.1f}%",
            "Status": "Warning" if metrics_b.bridges_count > metrics_a.bridges_count else "Nominal",
        },
        {
            "Category": "Cut-Set Analysis",
            "Metric": "Articulation Points (Cut-Vertices)",
            "Scenario A": f"{metrics_a.articulation_points_count}",
            "Scenario B": f"{metrics_b.articulation_points_count}",
            "Delta (B - A)": f"{metrics_b.articulation_points_count - metrics_a.articulation_points_count:+d}",
            "Change (%)": "N/A" if metrics_a.articulation_points_count == 0 else f"{((metrics_b.articulation_points_count - metrics_a.articulation_points_count) / metrics_a.articulation_points_count) * 100:+.1f}%",
            "Status": "Warning" if metrics_b.articulation_points_count > metrics_a.articulation_points_count else "Nominal",
        },
        {
            "Category": "Backbone Optimization",
            "Metric": "MST Total Resistance (Ω)",
            "Scenario A": f"{metrics_a.mst_resistance_ohm:.2f}",
            "Scenario B": f"{metrics_b.mst_resistance_ohm:.2f}",
            "Delta (B - A)": f"{metrics_b.mst_resistance_ohm - metrics_a.mst_resistance_ohm:+.2f}",
            "Change (%)": f"{((metrics_b.mst_resistance_ohm - metrics_a.mst_resistance_ohm) / max(0.01, metrics_a.mst_resistance_ohm)) * 100:+.1f}%",
            "Status": "Nominal",
        },
    ]
    return pd.DataFrame(rows)
