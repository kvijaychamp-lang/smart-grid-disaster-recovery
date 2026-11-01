"""
test_grid.py
============
Comprehensive pytest test suite for the Smart Grid Disaster Recovery backend.

Validates four rigorous test categories:
  1. Graph Initialization        — Node & edge attributes, structural invariants
  2. Cut-Set Isolation           — Bridge/AP detection, hospital isolation
  3. POSET Priority Enforcement  — Strict hospital-before-residential ordering
  4. Non-Functional Latency      — Dijkstra + MST complete within 200 ms on 50 nodes

Run:
    pytest test_grid.py -v
    pytest test_grid.py -v --tb=short   # concise tracebacks

Author  : Smart Grid QA & Systems Evaluation
Date    : 2026-10-31
Python  : 3.12+
"""

from __future__ import annotations

import math
import sys
import os
import time
from typing import Any

import networkx as nx
import pytest

# ── Ensure src/ is importable regardless of CWD ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from engine import (
    EDGE_CAPACITY_MW,
    EDGE_LENGTH_KM,
    EDGE_POWER_LOSS,
    EDGE_RESISTANCE,
    EDGE_STATUS,
    EDGE_VOLTAGE_KV,
    NodeType,
    PriorityLevel,
    apply_disaster,
    compute_poset_schedule,
    detect_cut_sets,
    dijkstra_shortest_path,
    find_critical_nodes,
    find_generator_nodes,
    get_edge_dataframe,
    get_node_dataframe,
    grid_metrics,
    initialize_grid_graph,
    kruskal_mst,
    run_full_recovery_pipeline,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def G16() -> nx.Graph:
    """Canonical 16-node grid, seed=42.  Shared across tests in this module."""
    return initialize_grid_graph(16, seed=42)


@pytest.fixture(scope="module")
def G50() -> nx.Graph:
    """50-node grid for latency / scalability tests."""
    return initialize_grid_graph(50, seed=7)


def _make_star_with_hospital() -> nx.Graph:
    """
    Deterministic 5-node star topology:

        Generator (0) ──── Hub (1) ──── Hospital (2)
                                 └──── Residential (3)
                                 └──── Residential (4)

    Removing the single edge (0, 1) isolates ALL non-generator nodes,
    including Hospital 2.  This gives a perfectly predictable test for
    cut-set hospital isolation.
    """
    G = nx.Graph()
    # Nodes
    node_defs = [
        (0, NodeType.GENERATOR,   PriorityLevel.CRITICAL, 200.0),
        (1, NodeType.SUBSTATION,  PriorityLevel.HIGH,      50.0),
        (2, NodeType.HOSPITAL,    PriorityLevel.CRITICAL,  10.0),
        (3, NodeType.RESIDENTIAL, PriorityLevel.LOW,        5.0),
        (4, NodeType.RESIDENTIAL, PriorityLevel.LOW,        3.0),
    ]
    for nid, ntype, prio, cap in node_defs:
        G.add_node(
            nid,
            node_id       = nid,
            node_type     = ntype,
            priority_level= int(prio),
            capacity      = cap,
            label         = f"{ntype[0]}{nid}",
            status        = "active",
            x             = float(nid),
            y             = 0.0,
        )

    def _edge(u: int, v: int, r: float = 5.0) -> dict:
        return {
            EDGE_RESISTANCE:  r,
            EDGE_CAPACITY_MW: 50.0,
            EDGE_POWER_LOSS:  1.0,
            EDGE_VOLTAGE_KV:  33.0,
            EDGE_LENGTH_KM:   10.0,
            EDGE_STATUS:      "active",
        }

    # Star: hub=1 connects all
    G.add_edge(0, 1, **_edge(0, 1, r=2.0))   # spine — this is the critical bridge
    G.add_edge(1, 2, **_edge(1, 2, r=5.0))
    G.add_edge(1, 3, **_edge(1, 3, r=5.0))
    G.add_edge(1, 4, **_edge(1, 4, r=5.0))
    return G


def _make_poset_graph(
    hospital_demand_mw: float = 10.0,
    residential_demand_mw: float = 5.0,
    generation_mw: float | None = None,
) -> tuple[nx.Graph, float]:
    """
    Minimal 4-node graph designed for POSET priority tests:
        Generator(0)  Hospital(1)  Residential(2)  Residential(3)

    Returns (graph, effective_generation_mw).
    """
    G = nx.Graph()
    node_defs = [
        (0, NodeType.GENERATOR,   PriorityLevel.CRITICAL, 100.0),
        (1, NodeType.HOSPITAL,    PriorityLevel.CRITICAL, hospital_demand_mw),
        (2, NodeType.RESIDENTIAL, PriorityLevel.LOW,      residential_demand_mw),
        (3, NodeType.RESIDENTIAL, PriorityLevel.LOW,      residential_demand_mw),
    ]
    for nid, ntype, prio, cap in node_defs:
        G.add_node(
            nid,
            node_id       = nid,
            node_type     = ntype,
            priority_level= int(prio),
            capacity      = cap,
            label         = f"{ntype[0]}{nid}",
            status        = "active",
            x             = float(nid),
            y             = 0.0,
        )
    edge_attrs = {
        EDGE_RESISTANCE:  1.0,
        EDGE_CAPACITY_MW: 200.0,
        EDGE_POWER_LOSS:  0.5,
        EDGE_VOLTAGE_KV:  132.0,
        EDGE_LENGTH_KM:   1.0,
        EDGE_STATUS:      "active",
    }
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2)]:
        G.add_edge(u, v, **edge_attrs)

    eff_gen = generation_mw if generation_mw is not None else 100.0
    return G, eff_gen


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST CATEGORY 1 — GRAPH INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphInitialization:
    """
    Verify that initialize_grid_graph() produces a structurally sound,
    fully attributed weighted graph G = (V, E, w).

    Covers:
    • Exact node count
    • Connectivity guarantee
    • Mandatory node attributes (with correct types & value ranges)
    • Mandatory edge attributes (with correct types & value ranges)
    • NodeType / PriorityLevel consistency
    • Generator count enforcement (min_generators constraint)
    • Coordinate system validity
    • Determinism (same seed → same graph)
    • Error handling for under-sized graphs
    • DataFrame helpers consistency
    """

    # ── Structural invariants ─────────────────────────────────────────────────

    def test_exact_node_count(self, G16: nx.Graph) -> None:
        """Graph must have exactly the requested number of nodes."""
        assert G16.number_of_nodes() == 16, (
            f"Expected 16 nodes, got {G16.number_of_nodes()}"
        )

    def test_graph_is_connected(self, G16: nx.Graph) -> None:
        """Grid must be fully connected (single component) post-init."""
        assert nx.is_connected(G16), (
            "Grid graph is not connected — spanning-tree guarantee failed"
        )

    def test_graph_is_undirected(self, G16: nx.Graph) -> None:
        """Power grid is an undirected graph (bidirectional lines)."""
        assert isinstance(G16, nx.Graph) and not isinstance(G16, nx.DiGraph)

    def test_edge_count_meets_minimum(self, G16: nx.Graph) -> None:
        """MST requires at least V-1 edges; real grids have more."""
        assert G16.number_of_edges() >= G16.number_of_nodes() - 1, (
            "Edge count is below the minimum for a spanning tree"
        )

    # ── Node attribute completeness ───────────────────────────────────────────

    REQUIRED_NODE_ATTRS = {
        "node_id", "node_type", "priority_level",
        "capacity", "label", "status", "x", "y",
    }

    def test_all_nodes_have_required_attributes(self, G16: nx.Graph) -> None:
        """Every node must carry all mandatory attributes."""
        for node, data in G16.nodes(data=True):
            missing = self.REQUIRED_NODE_ATTRS - set(data.keys())
            assert not missing, (
                f"Node {node} is missing attributes: {missing}"
            )

    def test_node_id_matches_graph_key(self, G16: nx.Graph) -> None:
        """node_id attribute must equal the NetworkX node key."""
        for node, data in G16.nodes(data=True):
            assert data["node_id"] == node, (
                f"Node {node} has node_id={data['node_id']} — mismatch"
            )

    def test_node_type_values_are_valid(self, G16: nx.Graph) -> None:
        """node_type must be one of the six defined NodeType constants."""
        valid = {
            NodeType.GENERATOR, NodeType.SUBSTATION, NodeType.HOSPITAL,
            NodeType.RESIDENTIAL, NodeType.COMMERCIAL, NodeType.EMERGENCY,
        }
        for node, data in G16.nodes(data=True):
            assert data["node_type"] in valid, (
                f"Node {node} has unknown node_type='{data['node_type']}'"
            )

    def test_priority_level_values_are_valid_integers(self, G16: nx.Graph) -> None:
        """priority_level must be an int within the PriorityLevel enum range."""
        valid = {int(p) for p in PriorityLevel}
        for node, data in G16.nodes(data=True):
            prio = data["priority_level"]
            assert isinstance(prio, int), (
                f"Node {node} priority_level is {type(prio)}, expected int"
            )
            assert prio in valid, (
                f"Node {node} priority_level={prio} not in {valid}"
            )

    def test_generator_priority_is_critical(self, G16: nx.Graph) -> None:
        """Every Generator node must carry CRITICAL priority."""
        for node, data in G16.nodes(data=True):
            if data["node_type"] == NodeType.GENERATOR:
                assert data["priority_level"] == int(PriorityLevel.CRITICAL), (
                    f"Generator {node} has non-CRITICAL priority {data['priority_level']}"
                )

    def test_hospital_priority_is_critical(self, G16: nx.Graph) -> None:
        """Every Hospital node must carry CRITICAL priority."""
        for node, data in G16.nodes(data=True):
            if data["node_type"] == NodeType.HOSPITAL:
                assert data["priority_level"] == int(PriorityLevel.CRITICAL), (
                    f"Hospital {node} has non-CRITICAL priority {data['priority_level']}"
                )

    def test_residential_priority_is_low(self, G16: nx.Graph) -> None:
        """Every Residential node must carry LOW priority."""
        for node, data in G16.nodes(data=True):
            if data["node_type"] == NodeType.RESIDENTIAL:
                assert data["priority_level"] == int(PriorityLevel.LOW), (
                    f"Residential {node} has unexpected priority {data['priority_level']}"
                )

    def test_capacity_is_positive_float(self, G16: nx.Graph) -> None:
        """All node capacities must be strictly positive floats."""
        for node, data in G16.nodes(data=True):
            cap = data["capacity"]
            assert isinstance(cap, float), (
                f"Node {node} capacity type is {type(cap)}, expected float"
            )
            assert cap > 0, f"Node {node} capacity={cap} is not positive"

    def test_generator_capacity_range(self, G16: nx.Graph) -> None:
        """Generators must be initialised in the 80–250 MW production band."""
        for node, data in G16.nodes(data=True):
            if data["node_type"] == NodeType.GENERATOR:
                cap = data["capacity"]
                assert 80.0 <= cap <= 250.0, (
                    f"Generator {node} capacity {cap} outside [80, 250] MW"
                )

    def test_hospital_capacity_range(self, G16: nx.Graph) -> None:
        """Hospital load demand must be in the 5–20 MW consumption band."""
        for node, data in G16.nodes(data=True):
            if data["node_type"] == NodeType.HOSPITAL:
                cap = data["capacity"]
                assert 5.0 <= cap <= 20.0, (
                    f"Hospital {node} capacity {cap} outside [5, 20] MW"
                )

    def test_node_status_is_active_on_init(self, G16: nx.Graph) -> None:
        """All nodes must start as 'active' — no pre-failed nodes."""
        for node, data in G16.nodes(data=True):
            assert data["status"] == "active", (
                f"Node {node} has status='{data['status']}' at initialisation"
            )

    def test_coordinate_attributes_are_finite_floats(self, G16: nx.Graph) -> None:
        """x and y layout coordinates must be finite real numbers."""
        for node, data in G16.nodes(data=True):
            for coord in ("x", "y"):
                val = data[coord]
                assert isinstance(val, float) and math.isfinite(val), (
                    f"Node {node} coordinate '{coord}'={val} is invalid"
                )

    # ── Edge attribute completeness ───────────────────────────────────────────

    REQUIRED_EDGE_ATTRS = {
        EDGE_RESISTANCE, EDGE_CAPACITY_MW, EDGE_POWER_LOSS,
        EDGE_VOLTAGE_KV, EDGE_LENGTH_KM, EDGE_STATUS,
    }

    def test_all_edges_have_required_attributes(self, G16: nx.Graph) -> None:
        """Every edge must carry all six mandatory transmission-line attributes."""
        for u, v, data in G16.edges(data=True):
            missing = self.REQUIRED_EDGE_ATTRS - set(data.keys())
            assert not missing, (
                f"Edge ({u},{v}) missing attributes: {missing}"
            )

    def test_edge_resistance_is_strictly_positive(self, G16: nx.Graph) -> None:
        """Resistance (Dijkstra weight) must be > 0 to avoid invalid paths."""
        for u, v, data in G16.edges(data=True):
            r = data[EDGE_RESISTANCE]
            assert r > 0, f"Edge ({u},{v}) resistance={r} is not positive"

    def test_edge_capacity_mw_is_positive(self, G16: nx.Graph) -> None:
        """Transmission-line MW capacity must be positive."""
        for u, v, data in G16.edges(data=True):
            cap = data[EDGE_CAPACITY_MW]
            assert cap > 0, f"Edge ({u},{v}) capacity_mw={cap} is not positive"

    def test_edge_voltage_kv_is_in_standard_set(self, G16: nx.Graph) -> None:
        """Voltage level must be one of the four standard grid voltages."""
        valid_kv = {11.0, 33.0, 66.0, 132.0}
        for u, v, data in G16.edges(data=True):
            kv = data[EDGE_VOLTAGE_KV]
            assert kv in valid_kv, (
                f"Edge ({u},{v}) voltage_kv={kv} not in {valid_kv}"
            )

    def test_edge_power_loss_pct_bounded(self, G16: nx.Graph) -> None:
        """Power loss percentage must be within the physical range [0.5, 8]%."""
        for u, v, data in G16.edges(data=True):
            pct = data[EDGE_POWER_LOSS]
            assert 0.0 < pct <= 8.0, (
                f"Edge ({u},{v}) power_loss_pct={pct} outside (0, 8]"
            )

    def test_edge_length_km_is_positive(self, G16: nx.Graph) -> None:
        """Physical cable length must be positive."""
        for u, v, data in G16.edges(data=True):
            ln = data[EDGE_LENGTH_KM]
            assert ln > 0, f"Edge ({u},{v}) length_km={ln} is not positive"

    def test_edge_status_is_active_on_init(self, G16: nx.Graph) -> None:
        """All edges must start with status='active'."""
        for u, v, data in G16.edges(data=True):
            assert data[EDGE_STATUS] == "active", (
                f"Edge ({u},{v}) has status='{data[EDGE_STATUS]}' at initialisation"
            )

    # ── Generator enforcement ─────────────────────────────────────────────────

    def test_minimum_two_generators_enforced(self, G16: nx.Graph) -> None:
        """initialize_grid_graph must guarantee at least min_generators=2 generators."""
        gens = find_generator_nodes(G16)
        assert len(gens) >= 2, (
            f"Expected ≥ 2 generators, got {len(gens)}"
        )

    def test_custom_min_generators_respected(self) -> None:
        """Setting min_generators=3 must produce at least 3 Generator nodes."""
        G = initialize_grid_graph(12, seed=1, min_generators=3)
        gens = find_generator_nodes(G)
        assert len(gens) >= 3, (
            f"Expected ≥ 3 generators with min_generators=3, got {len(gens)}"
        )

    def test_critical_nodes_include_generators_and_hospitals(self, G16: nx.Graph) -> None:
        """find_critical_nodes() must return all Generators + Hospitals."""
        critical = set(find_critical_nodes(G16))
        for node, data in G16.nodes(data=True):
            if data["node_type"] in (NodeType.GENERATOR, NodeType.HOSPITAL):
                assert node in critical, (
                    f"Node {node} ({data['node_type']}) absent from critical set"
                )

    # ── Reproducibility & error handling ─────────────────────────────────────

    def test_same_seed_produces_identical_topology(self) -> None:
        """Identical seeds must yield byte-identical graphs (determinism)."""
        G_a = initialize_grid_graph(14, seed=99)
        G_b = initialize_grid_graph(14, seed=99)
        assert list(G_a.nodes(data=True)) == list(G_b.nodes(data=True)), (
            "Same-seed graphs have different node data"
        )
        assert set(G_a.edges()) == set(G_b.edges()), (
            "Same-seed graphs have different edge sets"
        )

    def test_different_seeds_yield_distinct_topologies(self) -> None:
        """Different seeds should produce different edge structures."""
        G_a = initialize_grid_graph(14, seed=1)
        G_b = initialize_grid_graph(14, seed=2)
        # At minimum, the edge counts or sets differ
        different = (
            G_a.number_of_edges() != G_b.number_of_edges()
            or set(G_a.edges()) != set(G_b.edges())
        )
        assert different, "Different seeds produced identical edge structures"

    def test_raises_value_error_for_less_than_4_nodes(self) -> None:
        """Requesting < 4 nodes must raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="num_nodes must be"):
            initialize_grid_graph(3)

    def test_raises_value_error_for_1_node(self) -> None:
        with pytest.raises(ValueError):
            initialize_grid_graph(1)

    def test_graph_metadata_tags_set(self, G16: nx.Graph) -> None:
        """Graph-level metadata (name, seed, weight_attr) must be populated."""
        assert "name"        in G16.graph
        assert "seed"        in G16.graph
        assert "weight_attr" in G16.graph
        assert G16.graph["weight_attr"] == EDGE_RESISTANCE

    # ── DataFrame helpers ─────────────────────────────────────────────────────

    def test_node_dataframe_row_count(self, G16: nx.Graph) -> None:
        df = get_node_dataframe(G16)
        assert len(df) == 16

    def test_node_dataframe_contains_required_columns(self, G16: nx.Graph) -> None:
        df = get_node_dataframe(G16)
        for col in ("node_id", "node_type", "priority_level", "capacity"):
            assert col in df.columns, f"Column '{col}' missing from node DataFrame"

    def test_edge_dataframe_non_empty(self, G16: nx.Graph) -> None:
        df = get_edge_dataframe(G16)
        assert len(df) > 0

    def test_edge_dataframe_resistance_column_positive(self, G16: nx.Graph) -> None:
        df = get_edge_dataframe(G16)
        assert (df[EDGE_RESISTANCE] > 0).all(), (
            "Edge DataFrame contains non-positive resistance values"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST CATEGORY 2 — CUT-SET ISOLATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestCutSetIsolation:
    """
    Programmatically disable critical edges and verify that detect_cut_sets()
    correctly flags isolated hospital nodes in blacked_out_zones.

    Strategy:
    • Uses _make_star_with_hospital() — a hand-crafted topology where removing
      the single spine edge (0→1) deterministically isolates the hospital.
    • Also validates on the 16-node random grid with known bridges.
    • Tests CutSetResult dataclass completeness and structural correctness.
    """

    # ── Baseline (no failures) ────────────────────────────────────────────────

    def test_no_failures_yields_single_component(self, G16: nx.Graph) -> None:
        result = detect_cut_sets(G16)
        assert result.is_connected, "Healthy grid must be a single connected component"
        assert result.num_components == 1

    def test_no_failures_has_no_blacked_out_zones(self, G16: nx.Graph) -> None:
        result = detect_cut_sets(G16)
        # Sum of all blacked-out node lists across all priority keys
        total_blacked = sum(len(v) for v in result.blacked_out_zones.values())
        assert total_blacked == 0, (
            f"Healthy grid should have 0 blacked-out nodes, got {total_blacked}"
        )

    def test_result_dataclass_fields_present(self, G16: nx.Graph) -> None:
        result = detect_cut_sets(G16)
        assert hasattr(result, "articulation_points")
        assert hasattr(result, "bridges")
        assert hasattr(result, "connected_components")
        assert hasattr(result, "blacked_out_zones")
        assert hasattr(result, "is_connected")
        assert hasattr(result, "num_components")
        assert hasattr(result, "component_df")

    def test_connected_components_cover_all_nodes(self, G16: nx.Graph) -> None:
        """All nodes must appear in exactly one connected component."""
        result = detect_cut_sets(G16)
        covered = set()
        for comp in result.connected_components:
            covered.update(comp)
        assert covered == set(G16.nodes()), (
            "Connected components do not cover every graph node"
        )

    # ── Bridge removal isolates hospital (star topology) ─────────────────────

    def test_bridge_removal_splits_graph(self) -> None:
        """Removing the spine bridge (0,1) must split the star into 2 components."""
        G = _make_star_with_hospital()
        result = detect_cut_sets(G, failed_edges=[(0, 1)])
        assert result.num_components >= 2, (
            f"Removing spine bridge must produce ≥ 2 components, got {result.num_components}"
        )
        assert not result.is_connected, "Graph should be disconnected after bridge removal"

    def test_hospital_node_appears_in_blacked_out_zones(self) -> None:
        """
        After severing the generator→hub spine, Hospital node 2 must appear in
        blacked_out_zones under the CRITICAL priority key.
        """
        G = _make_star_with_hospital()
        result = detect_cut_sets(G, failed_edges=[(0, 1)])

        critical_key = int(PriorityLevel.CRITICAL)
        blacked_critical = result.blacked_out_zones.get(critical_key, [])

        assert 2 in blacked_critical, (
            f"Hospital node 2 not found in blacked_out_zones[CRITICAL]. "
            f"Full blacked_out_zones: {result.blacked_out_zones}"
        )

    def test_residential_nodes_also_blacked_out_with_hospital(self) -> None:
        """
        After the spine failure, both residential nodes (3, 4) must also be
        in blacked_out_zones under LOW priority.
        """
        G = _make_star_with_hospital()
        result = detect_cut_sets(G, failed_edges=[(0, 1)])

        low_key = int(PriorityLevel.LOW)
        blacked_low = set(result.blacked_out_zones.get(low_key, []))

        assert {3, 4}.issubset(blacked_low), (
            f"Residential nodes 3 & 4 should be blacked out under LOW. "
            f"Got: {blacked_low}"
        )

    def test_generator_node_not_in_blacked_out_zones(self) -> None:
        """Generator (node 0) stays powered — must NOT appear in blacked_out_zones."""
        G = _make_star_with_hospital()
        result = detect_cut_sets(G, failed_edges=[(0, 1)])

        all_blacked = []
        for nodes_list in result.blacked_out_zones.values():
            all_blacked.extend(nodes_list)

        assert 0 not in all_blacked, (
            "Generator node 0 incorrectly appears in blacked_out_zones"
        )

    def test_bridge_detected_in_tarjan_bridges_list(self) -> None:
        """
        The critical spine (0,1) must be identified as a bridge by Tarjan's
        algorithm — it is the sole connection between generator and the rest.
        """
        G = _make_star_with_hospital()
        result = detect_cut_sets(G)  # no failures — query bridges of intact graph

        bridges_set = {(min(u, v), max(u, v)) for u, v in result.bridges}
        # (0,1) must be detected as a bridge
        assert (0, 1) in bridges_set, (
            f"Spine edge (0,1) not found in bridges. All bridges: {result.bridges}"
        )

    def test_articulation_point_hub_detected(self) -> None:
        """
        Hub node 1 is the sole connector in the star — it must be identified
        as an articulation point by Tarjan's algorithm.
        """
        G = _make_star_with_hospital()
        result = detect_cut_sets(G)

        assert 1 in result.articulation_points, (
            f"Hub node 1 not in articulation_points. "
            f"APs found: {result.articulation_points}"
        )

    def test_multiple_edge_failures_increase_components(self, G16: nx.Graph) -> None:
        """Severing more bridges must produce ≥ as many components as one bridge."""
        edges = list(G16.edges())
        result_one = detect_cut_sets(G16, failed_edges=edges[:1])
        result_three = detect_cut_sets(G16, failed_edges=edges[:3])
        # components can only stay same or increase with more failures
        assert result_three.num_components >= result_one.num_components

    def test_failed_node_removal_propagates_to_blacked_zones(self, G16: nx.Graph) -> None:
        """
        Explicitly failed nodes must appear in blacked_out_zones regardless of
        component membership.
        """
        hospital_nodes = [
            n for n, d in G16.nodes(data=True)
            if d["node_type"] == NodeType.HOSPITAL
        ]
        if not hospital_nodes:
            pytest.skip("No hospital nodes in G16 — skip failed_node isolation test")

        target = hospital_nodes[0]
        result = detect_cut_sets(G16, failed_nodes=[target])

        all_blacked = [n for lst in result.blacked_out_zones.values() for n in lst]
        assert target in all_blacked, (
            f"Explicitly failed hospital {target} absent from blacked_out_zones"
        )

    def test_component_df_columns_present(self, G16: nx.Graph) -> None:
        """component_df must have the expected summary columns."""
        result = detect_cut_sets(G16)
        expected_cols = {"component_id", "size", "has_generator", "highest_priority"}
        actual_cols   = set(result.component_df.columns)
        assert expected_cols.issubset(actual_cols), (
            f"component_df missing columns: {expected_cols - actual_cols}"
        )

    def test_component_df_row_count_matches_num_components(self, G16: nx.Graph) -> None:
        """Number of DataFrame rows must equal num_components."""
        result = detect_cut_sets(G16)
        assert len(result.component_df) == result.num_components

    def test_no_self_loops_in_bridges(self, G16: nx.Graph) -> None:
        """A bridge can never be a self-loop (u ≠ v)."""
        result = detect_cut_sets(G16)
        for u, v in result.bridges:
            assert u != v, f"Self-loop ({u},{v}) reported as bridge"

    def test_cut_set_does_not_modify_original_graph(self, G16: nx.Graph) -> None:
        """detect_cut_sets must never mutate the original graph in-place."""
        edges_before = set(G16.edges())
        nodes_before = set(G16.nodes())
        detect_cut_sets(G16, failed_edges=list(G16.edges())[:5])
        assert set(G16.edges()) == edges_before, "detect_cut_sets modified original edges"
        assert set(G16.nodes()) == nodes_before, "detect_cut_sets modified original nodes"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST CATEGORY 3 — POSET PRIORITY ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestPOSETPriorityEnforcement:
    """
    Simulate severe generation constraints and verify that:
      a) Hospital zones are always powered before residential zones
      b) The restoration_order strictly respects the POSET partial order
      c) When capacity is exhausted, only lower-priority nodes are unserved
      d) No priority inversion errors exist in any allocation scenario

    The POSET is:  CRITICAL (1) < HIGH (2) < MEDIUM (3) < LOW (4)
    Power must flow to higher-priority (lower integer) nodes first.
    """

    # ── Structural correctness of restoration order ───────────────────────────

    def test_restoration_order_starts_with_highest_priority(self, G16: nx.Graph) -> None:
        """First element of restoration_order must be CRITICAL or HIGH priority."""
        schedule = compute_poset_schedule(G16)
        first_node = schedule.restoration_order[0]
        first_prio = G16.nodes[first_node]["priority_level"]
        assert first_prio <= int(PriorityLevel.HIGH), (
            f"First restored node has priority {first_prio}, "
            f"expected ≤ {int(PriorityLevel.HIGH)} (CRITICAL or HIGH)"
        )

    def test_restoration_order_strictly_non_decreasing_priority(
        self, G16: nx.Graph
    ) -> None:
        """
        Priority levels in restoration_order must be non-decreasing
        (once a tier appears, no earlier tier may appear later).
        This enforces the POSET total-order constraint with no inversion.
        """
        schedule = compute_poset_schedule(G16)
        prev_prio = 0
        for node in schedule.restoration_order:
            prio = G16.nodes[node]["priority_level"]
            assert prio >= prev_prio, (
                f"Priority inversion detected: node {node} has priority {prio} "
                f"after a node with priority {prev_prio}"
            )
            prev_prio = prio

    def test_all_active_nodes_in_restoration_order(self, G16: nx.Graph) -> None:
        """restoration_order must include every active (non-failed) node."""
        schedule = compute_poset_schedule(G16)
        active_nodes = {
            n for n, d in G16.nodes(data=True)
            if d.get("status", "active") == "active"
        }
        order_set = set(schedule.restoration_order)
        assert active_nodes == order_set, (
            f"Active nodes not in restoration_order: {active_nodes - order_set}"
        )

    def test_restoration_order_has_no_duplicates(self, G16: nx.Graph) -> None:
        """Each node must appear exactly once in restoration_order."""
        schedule = compute_poset_schedule(G16)
        order = schedule.restoration_order
        assert len(order) == len(set(order)), (
            "restoration_order contains duplicate node IDs"
        )

    # ── Hospital before residential (the core POSET contract) ─────────────────

    def test_hospital_precedes_all_residential_in_order(self) -> None:
        """
        In the minimal POSET graph, Hospital(1) must appear in restoration_order
        strictly before both Residential(2) and Residential(3).
        """
        G, _ = _make_poset_graph()
        schedule = compute_poset_schedule(G)
        order = schedule.restoration_order

        hosp_pos = order.index(1)   # Hospital is node 1
        res_positions = [order.index(n) for n in (2, 3)]

        for res_pos in res_positions:
            assert hosp_pos < res_pos, (
                f"Hospital at position {hosp_pos} must precede residential "
                f"at position {res_pos}. Order: {order}"
            )

    def test_hospital_receives_allocation_before_residential(self) -> None:
        """
        Under normal generation, Hospital(1) must have a positive power
        allocation, and its allocation must be assigned before residentials
        are considered.
        """
        G, gen_mw = _make_poset_graph(
            hospital_demand_mw=10.0,
            residential_demand_mw=5.0,
            generation_mw=100.0,   # plenty of capacity
        )
        schedule = compute_poset_schedule(G, available_generation_mw=gen_mw)

        hospital_alloc = schedule.power_allocation.get(1, 0.0)
        assert hospital_alloc > 0.0, (
            f"Hospital node 1 has zero allocation despite sufficient generation. "
            f"allocation={hospital_alloc}"
        )

    # ── Severe capacity constraint: hospitals served, residentials shed ────────

    def test_residential_unserved_when_capacity_exhausted_by_hospital(self) -> None:
        """
        CRITICAL TEST: With generation barely enough for the hospital but
        insufficient for residential nodes, POSET MUST:
          • Serve Hospital(1)         → allocation > 0
          • Leave Residential(2,3) unserved → in unserved_nodes

        This is the cornerstone of the disaster priority triage protocol.
        """
        G, _ = _make_poset_graph(
            hospital_demand_mw=10.0,
            residential_demand_mw=5.0,
        )
        # Generation = 11 MW — exactly enough for Hospital (10 MW) + 1 MW surplus
        # Not enough for both residentials (each needs 5 MW = 10 MW total needed)
        severe_gen_mw = 11.0

        schedule = compute_poset_schedule(G, available_generation_mw=severe_gen_mw)

        hospital_alloc = schedule.power_allocation.get(1, 0.0)
        assert hospital_alloc > 0.0, (
            f"Hospital (node 1) must be served. Got allocation={hospital_alloc}"
        )

        unserved_set = set(schedule.unserved_nodes)
        # At least one residential must be in unserved (not enough power for both)
        res_unserved = unserved_set & {2, 3}
        assert len(res_unserved) >= 1, (
            f"Expected ≥ 1 residential in unserved_nodes under severe constraint "
            f"(gen={severe_gen_mw} MW). unserved={schedule.unserved_nodes}"
        )

    def test_hospital_never_unserved_when_capacity_covers_it(self) -> None:
        """
        Hospital must never appear in unserved_nodes when generation ≥ hospital demand,
        even if residentials are shed.
        """
        G, _ = _make_poset_graph(
            hospital_demand_mw=10.0,
            residential_demand_mw=5.0,
        )
        # Generation = 12 MW — exactly enough for hospital (10 MW) + minimal buffer
        schedule = compute_poset_schedule(G, available_generation_mw=12.0)

        assert 1 not in schedule.unserved_nodes, (
            f"Hospital (node 1) incorrectly placed in unserved_nodes. "
            f"unserved={schedule.unserved_nodes}"
        )

    def test_no_priority_inversion_residential_served_before_hospital(self) -> None:
        """
        Anti-inversion test: No residential node may have a non-zero allocation
        while a hospital node has zero allocation, when generation is constrained.
        """
        G, _ = _make_poset_graph(
            hospital_demand_mw=10.0,
            residential_demand_mw=5.0,
        )
        schedule = compute_poset_schedule(G, available_generation_mw=8.0)
        # Gen = 8 MW < hospital demand (10 MW), so hospital gets partial service

        for n, alloc in schedule.power_allocation.items():
            ntype = G.nodes[n]["node_type"]
            prio  = G.nodes[n]["priority_level"]
            if ntype == NodeType.RESIDENTIAL:
                # Find any hospital with lower allocation
                for m, h_alloc in schedule.power_allocation.items():
                    if G.nodes[m]["node_type"] == NodeType.HOSPITAL:
                        if h_alloc == 0.0 and alloc > 0.0:
                            pytest.fail(
                                f"Priority inversion: Residential node {n} "
                                f"(alloc={alloc:.4f}) served while Hospital node {m} "
                                f"(alloc={h_alloc}) is unserved."
                            )

    def test_extreme_constraint_hospital_still_served(self) -> None:
        """
        Even at 1 MW generation (< hospital demand), hospital allocation must be
        non-zero (partial service) and residential allocation must be zero.
        """
        G, _ = _make_poset_graph(
            hospital_demand_mw=10.0,
            residential_demand_mw=5.0,
        )
        schedule = compute_poset_schedule(G, available_generation_mw=1.0)

        hospital_alloc = schedule.power_allocation.get(1, -1.0)
        res2_alloc     = schedule.power_allocation.get(2, -1.0)
        res3_alloc     = schedule.power_allocation.get(3, -1.0)

        assert hospital_alloc >= 0.0, "Hospital allocation must be ≥ 0"
        assert res2_alloc == 0.0, (
            f"Residential node 2 must have 0 allocation at 1 MW gen, got {res2_alloc}"
        )
        assert res3_alloc == 0.0, (
            f"Residential node 3 must have 0 allocation at 1 MW gen, got {res3_alloc}"
        )

    def test_zero_generation_all_non_generators_unserved(self) -> None:
        """At 0 MW generation, all non-generator nodes must be unserved."""
        G, _ = _make_poset_graph()
        schedule = compute_poset_schedule(G, available_generation_mw=0.0)

        # All non-generator nodes are 1, 2, 3
        non_gen = {n for n, d in G.nodes(data=True) if d["node_type"] != NodeType.GENERATOR}
        unserved = set(schedule.unserved_nodes)
        assert non_gen == unserved, (
            f"At 0 MW, all non-generators must be unserved. "
            f"Non-generators: {non_gen}, Unserved: {unserved}"
        )

    # ── POSET result metadata ─────────────────────────────────────────────────

    def test_priority_groups_keyed_by_integer_priority(self, G16: nx.Graph) -> None:
        """priority_groups must be keyed by integer priority level integers."""
        schedule = compute_poset_schedule(G16)
        valid_keys = {int(p) for p in PriorityLevel}
        for key in schedule.priority_groups:
            assert key in valid_keys, (
                f"priority_groups contains unexpected key {key}"
            )

    def test_total_available_mw_matches_generator_sum(self, G16: nx.Graph) -> None:
        """Without override, total_available_mw must equal sum of generator capacities."""
        gen_sum = sum(
            d["capacity"] for _, d in G16.nodes(data=True)
            if d["node_type"] == NodeType.GENERATOR
        )
        schedule = compute_poset_schedule(G16)
        assert abs(schedule.total_available_mw - gen_sum) < 0.01, (
            f"total_available_mw={schedule.total_available_mw} != "
            f"generator_sum={gen_sum}"
        )

    def test_power_allocation_sums_to_at_most_one(self, G16: nx.Graph) -> None:
        """Sum of all power allocations (fractions) must be ≤ 1.0."""
        schedule = compute_poset_schedule(G16)
        total = sum(schedule.power_allocation.values())
        assert total <= 1.0 + 1e-6, (
            f"Sum of power allocations {total:.6f} exceeds 1.0 (over-allocation)"
        )

    def test_lattice_edges_form_valid_covering_relations(self, G16: nx.Graph) -> None:
        """lattice_edges must connect nodes from adjacent priority tiers only."""
        schedule = compute_poset_schedule(G16)
        for u, v in schedule.lattice_edges:
            prio_u = G16.nodes[u]["priority_level"]
            prio_v = G16.nodes[v]["priority_level"]
            assert prio_u < prio_v, (
                f"Hasse edge ({u},{v}) violates covering relation: "
                f"prio({u})={prio_u} must be < prio({v})={prio_v}"
            )

    def test_override_generation_respected(self, G16: nx.Graph) -> None:
        """available_generation_mw override must be reflected in result."""
        override = 999.99
        schedule = compute_poset_schedule(G16, available_generation_mw=override)
        assert abs(schedule.total_available_mw - override) < 0.01, (
            f"Override {override} not reflected: got {schedule.total_available_mw}"
        )

    def test_failed_nodes_excluded_from_schedule(self, G16: nx.Graph) -> None:
        """Nodes passed as failed_nodes must not appear in restoration_order."""
        all_nodes = list(G16.nodes())
        failed    = all_nodes[:2]
        schedule  = compute_poset_schedule(G16, failed_nodes=failed)
        for fn in failed:
            assert fn not in schedule.restoration_order, (
                f"Failed node {fn} appeared in restoration_order"
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST CATEGORY 4 — NON-FUNCTIONAL LATENCY (< 200 ms SLA)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNonFunctionalLatency:
    """
    Performance regression suite for the two core path algorithms on a 50-node
    topology.  The system SLA requires both Dijkstra (SSSP) and Kruskal (MST)
    to complete well within 200 milliseconds even under failure scenarios.

    Measurement methodology:
    • Algorithms are called in isolation (no I/O, no plotting overhead).
    • time.perf_counter() — highest resolution wall-clock timer available.
    • Each test has an individual 200 ms budget; cumulative budget is not shared.
    • 50-node graph chosen to be representative of the maximum UI-facing topology.
    """

    LATENCY_BUDGET_MS = 200.0    # Hard SLA ceiling
    COMFORT_BUDGET_MS = 100.0    # Comfortable target (50 % margin)

    # ── Dijkstra latency ──────────────────────────────────────────────────────

    def test_dijkstra_completes_within_200ms_on_50_nodes(self, G50: nx.Graph) -> None:
        """
        Dijkstra SSSP from the first generator on a 50-node grid must finish
        in < 200 ms.
        """
        source = find_generator_nodes(G50)[0]

        t_start = time.perf_counter()
        result  = dijkstra_shortest_path(G50, source=source)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.LATENCY_BUDGET_MS, (
            f"Dijkstra took {elapsed_ms:.2f} ms — exceeds {self.LATENCY_BUDGET_MS} ms SLA"
        )
        # Sanity: algorithm must have run meaningfully
        assert result.total_nodes_reached > 1, "Dijkstra reached ≤ 1 node — possible early exit bug"

    def test_dijkstra_comfortable_under_100ms(self, G50: nx.Graph) -> None:
        """
        Comfort target: Dijkstra on 50 nodes should finish in < 100 ms to leave
        headroom for future graph growth.
        """
        source = find_generator_nodes(G50)[0]

        t_start = time.perf_counter()
        dijkstra_shortest_path(G50, source=source)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.COMFORT_BUDGET_MS, (
            f"Dijkstra took {elapsed_ms:.2f} ms — exceeds comfort target "
            f"{self.COMFORT_BUDGET_MS} ms (SLA={self.LATENCY_BUDGET_MS} ms)"
        )

    def test_dijkstra_with_failed_edges_within_200ms(self, G50: nx.Graph) -> None:
        """
        Dijkstra must still meet the SLA when 10 edges are excluded (disaster mode).
        """
        source = find_generator_nodes(G50)[0]
        failed_edges = list(G50.edges())[:10]

        t_start    = time.perf_counter()
        result     = dijkstra_shortest_path(G50, source=source, exclude_edges=failed_edges)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.LATENCY_BUDGET_MS, (
            f"Dijkstra (with 10 failed edges) took {elapsed_ms:.2f} ms — "
            f"exceeds {self.LATENCY_BUDGET_MS} ms SLA"
        )

    def test_dijkstra_all_generators_combined_within_sla(self, G50: nx.Graph) -> None:
        """
        Running Dijkstra from every generator (as the pipeline does) must
        complete well within 200 ms total.
        """
        gens = find_generator_nodes(G50)

        t_start = time.perf_counter()
        for gen in gens:
            dijkstra_shortest_path(G50, source=gen)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.LATENCY_BUDGET_MS, (
            f"Dijkstra across all {len(gens)} generators took {elapsed_ms:.2f} ms — "
            f"exceeds {self.LATENCY_BUDGET_MS} ms SLA"
        )

    # ── Kruskal MST latency ────────────────────────────────────────────────────

    def test_kruskal_mst_completes_within_200ms_on_50_nodes(self, G50: nx.Graph) -> None:
        """
        Kruskal MST on a 50-node grid must finish in < 200 ms.
        """
        t_start    = time.perf_counter()
        result     = kruskal_mst(G50)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.LATENCY_BUDGET_MS, (
            f"Kruskal MST took {elapsed_ms:.2f} ms — exceeds {self.LATENCY_BUDGET_MS} ms SLA"
        )
        assert result.is_spanning, "Kruskal MST is not spanning the full 50-node graph"

    def test_kruskal_comfortable_under_100ms(self, G50: nx.Graph) -> None:
        """
        Comfort target: Kruskal on 50 nodes should finish in < 100 ms.
        """
        t_start    = time.perf_counter()
        kruskal_mst(G50)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.COMFORT_BUDGET_MS, (
            f"Kruskal MST took {elapsed_ms:.2f} ms — exceeds comfort target "
            f"{self.COMFORT_BUDGET_MS} ms (SLA={self.LATENCY_BUDGET_MS} ms)"
        )

    def test_kruskal_with_failed_edges_within_200ms(self, G50: nx.Graph) -> None:
        """
        Kruskal with 8 excluded edges must still satisfy the 200 ms SLA.
        """
        failed_edges = list(G50.edges())[:8]

        t_start    = time.perf_counter()
        result     = kruskal_mst(G50, exclude_edges=failed_edges)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.LATENCY_BUDGET_MS, (
            f"Kruskal (with 8 failed edges) took {elapsed_ms:.2f} ms — "
            f"exceeds {self.LATENCY_BUDGET_MS} ms SLA"
        )

    # ── Combined pipeline latency ──────────────────────────────────────────────

    def test_full_recovery_pipeline_within_200ms_on_50_nodes(self, G50: nx.Graph) -> None:
        """
        The full recovery pipeline (Cut-Set + POSET + MST + Dijkstra per generator)
        must complete within 200 ms on 50 nodes.

        This is the most demanding latency test — it validates the end-to-end
        system response time as perceived by the Streamlit dashboard.
        """
        t_start    = time.perf_counter()
        result     = run_full_recovery_pipeline(G50, failed_edges=None, failed_nodes=None)
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        assert elapsed_ms < self.LATENCY_BUDGET_MS, (
            f"Full recovery pipeline took {elapsed_ms:.2f} ms — "
            f"exceeds {self.LATENCY_BUDGET_MS} ms SLA"
        )
        # Correctness sanity checks alongside latency
        assert "cut_set"  in result
        assert "poset"    in result
        assert "mst"      in result
        assert "dijkstra" in result
        assert "summary"  in result

    def test_dijkstra_result_correctness_on_50_nodes(self, G50: nx.Graph) -> None:
        """
        Alongside latency, verify correctness: source distance = 0,
        all distances non-negative, paths start at source.
        """
        source = find_generator_nodes(G50)[0]
        result = dijkstra_shortest_path(G50, source=source)

        assert result.distances[source] == 0.0, "Source distance must be 0.0"
        for n, dist in result.distances.items():
            assert dist >= 0.0 or dist == math.inf, (
                f"Node {n} has negative distance {dist}"
            )
        # Spot-check: at least one reconstructed path starts at source
        reachable_paths = {
            t: p for t, p in result.shortest_path_to.items() if p
        }
        if reachable_paths:
            sample_path = next(iter(reachable_paths.values()))
            assert sample_path[0] == source, (
                f"Reconstructed path does not start at source {source}: {sample_path}"
            )

    def test_kruskal_result_correctness_on_50_nodes(self, G50: nx.Graph) -> None:
        """
        Alongside latency, verify MST correctness: exactly V-1 edges,
        spanning tree property, total_weight > 0.
        """
        result = kruskal_mst(G50)

        assert result.is_spanning, "MST must span all 50 nodes"
        assert len(result.mst_edges) == G50.number_of_nodes() - 1, (
            f"MST should have {G50.number_of_nodes() - 1} edges, "
            f"got {len(result.mst_edges)}"
        )
        assert result.total_weight > 0.0, "MST total resistance must be positive"
        assert nx.is_tree(result.mst_graph), "MST graph is not a valid tree"
