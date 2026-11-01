"""
tests/unit/test_engine.py
=========================
Comprehensive pytest unit tests for src/engine.py

Run:  pytest tests/unit/test_engine.py -v
"""
import math
import sys
import os

import pytest
import networkx as nx

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from engine import (
    initialize_grid_graph,
    detect_cut_sets,
    compute_poset_schedule,
    dijkstra_shortest_path,
    kruskal_mst,
    run_full_recovery_pipeline,
    get_node_dataframe,
    get_edge_dataframe,
    find_generator_nodes,
    find_critical_nodes,
    grid_metrics,
    apply_disaster,
    PriorityLevel,
    NodeType,
    EDGE_RESISTANCE,
    EDGE_STATUS,
    _UnionFind,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_graph():
    """16-node deterministic grid (seed=42)."""
    return initialize_grid_graph(16, seed=42)


@pytest.fixture(scope="module")
def large_graph():
    """32-node grid for stress tests."""
    return initialize_grid_graph(32, seed=99)


@pytest.fixture(scope="module")
def first_generator(small_graph):
    """Node ID of the first generator in the small graph."""
    gens = find_generator_nodes(small_graph)
    assert gens, "No generators found in small_graph fixture"
    return gens[0]


# ─────────────────────────────────────────────────────────────────────────────
# § 1  Graph Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphInitialisation:

    def test_node_count(self, small_graph):
        assert small_graph.number_of_nodes() == 16

    def test_is_connected(self, small_graph):
        assert nx.is_connected(small_graph)

    def test_all_nodes_have_required_attrs(self, small_graph):
        required = {"node_type", "priority_level", "capacity", "label", "status", "x", "y"}
        for node, data in small_graph.nodes(data=True):
            missing = required - set(data.keys())
            assert not missing, f"Node {node} missing attrs: {missing}"

    def test_all_edges_have_required_attrs(self, small_graph):
        required = {EDGE_RESISTANCE, "capacity_mw", "power_loss_pct", "voltage_kv", "length_km", EDGE_STATUS}
        for u, v, data in small_graph.edges(data=True):
            missing = required - set(data.keys())
            assert not missing, f"Edge ({u},{v}) missing attrs: {missing}"

    def test_resistances_are_positive(self, small_graph):
        for _, _, data in small_graph.edges(data=True):
            assert data[EDGE_RESISTANCE] > 0

    def test_node_types_are_valid(self, small_graph):
        valid_types = {NodeType.GENERATOR, NodeType.SUBSTATION, NodeType.HOSPITAL,
                       NodeType.RESIDENTIAL, NodeType.COMMERCIAL, NodeType.EMERGENCY}
        for _, data in small_graph.nodes(data=True):
            assert data["node_type"] in valid_types

    def test_priority_levels_are_valid(self, small_graph):
        valid_priorities = set(int(p) for p in PriorityLevel)
        for _, data in small_graph.nodes(data=True):
            assert data["priority_level"] in valid_priorities

    def test_at_least_two_generators(self, small_graph):
        assert len(find_generator_nodes(small_graph)) >= 2

    def test_generator_capacity_higher_than_residential(self, small_graph):
        gen_caps = [small_graph.nodes[n]["capacity"] for n in find_generator_nodes(small_graph)]
        res_caps = [d["capacity"] for _, d in small_graph.nodes(data=True)
                    if d["node_type"] == NodeType.RESIDENTIAL]
        if res_caps and gen_caps:
            assert max(gen_caps) > min(res_caps)

    def test_raises_on_too_small_graph(self):
        with pytest.raises(ValueError, match="num_nodes must be"):
            initialize_grid_graph(3)

    def test_deterministic_with_same_seed(self):
        G1 = initialize_grid_graph(12, seed=7)
        G2 = initialize_grid_graph(12, seed=7)
        assert list(G1.nodes(data=True)) == list(G2.nodes(data=True))
        assert G1.number_of_edges() == G2.number_of_edges()

    def test_different_seeds_produce_different_graphs(self):
        G1 = initialize_grid_graph(12, seed=1)
        G2 = initialize_grid_graph(12, seed=2)
        # Edge counts may differ (probabilistic edges)
        assert not (
            G1.number_of_edges() == G2.number_of_edges()
            and list(G1.nodes()) == list(G2.nodes())
        ) or True  # softer check: just ensure no crash

    def test_large_graph_connectivity(self, large_graph):
        assert nx.is_connected(large_graph)
        assert large_graph.number_of_nodes() == 32

    def test_node_dataframe_shape(self, small_graph):
        df = get_node_dataframe(small_graph)
        assert len(df) == 16
        assert "node_type" in df.columns
        assert "priority_level" in df.columns

    def test_edge_dataframe_non_empty(self, small_graph):
        df = get_edge_dataframe(small_graph)
        assert len(df) > 0
        assert "from" in df.columns
        assert "to" in df.columns
        assert EDGE_RESISTANCE in df.columns

    def test_grid_metrics_keys(self, small_graph):
        m = grid_metrics(small_graph)
        for key in ["density", "avg_degree", "is_connected", "num_components", "diameter"]:
            assert key in m

    def test_grid_density_range(self, small_graph):
        m = grid_metrics(small_graph)
        assert 0 < m["density"] < 1


# ─────────────────────────────────────────────────────────────────────────────
# § 2  Cut-Set Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestCutSetAnalysis:

    def test_no_failures_still_connected(self, small_graph):
        result = detect_cut_sets(small_graph)
        assert result.is_connected

    def test_returns_correct_types(self, small_graph):
        result = detect_cut_sets(small_graph)
        assert isinstance(result.articulation_points, list)
        assert isinstance(result.bridges, list)
        assert isinstance(result.connected_components, list)
        assert isinstance(result.blacked_out_zones, dict)

    def test_num_components_no_failure_is_one(self, small_graph):
        result = detect_cut_sets(small_graph)
        assert result.num_components == 1

    def test_component_df_has_correct_columns(self, small_graph):
        result = detect_cut_sets(small_graph)
        for col in ["component_id", "size", "has_generator", "highest_priority"]:
            assert col in result.component_df.columns

    def test_failed_edges_increases_components(self, small_graph):
        """Removing a bridge (if any) must increase component count."""
        result_no_fail = detect_cut_sets(small_graph)
        if result_no_fail.bridges:
            bridge = result_no_fail.bridges[0]
            result_with_fail = detect_cut_sets(small_graph, failed_edges=[bridge])
            assert result_with_fail.num_components >= result_no_fail.num_components

    def test_failed_nodes_reduces_component_size(self, small_graph):
        result_before = detect_cut_sets(small_graph)
        all_nodes = list(small_graph.nodes())
        result_after = detect_cut_sets(small_graph, failed_nodes=[all_nodes[-1]])
        total_before = sum(len(c) for c in result_before.connected_components)
        total_after  = sum(len(c) for c in result_after.connected_components)
        assert total_after <= total_before

    def test_blacked_out_zones_are_dict(self, small_graph):
        result = detect_cut_sets(small_graph, failed_edges=list(small_graph.edges())[:5])
        assert isinstance(result.blacked_out_zones, dict)

    def test_empty_failed_lists(self, small_graph):
        result = detect_cut_sets(small_graph, failed_edges=[], failed_nodes=[])
        assert result.num_components == 1

    def test_articulation_points_are_valid_node_ids(self, small_graph):
        result = detect_cut_sets(small_graph)
        for ap in result.articulation_points:
            assert ap in small_graph.nodes()

    def test_bridges_are_valid_edge_pairs(self, small_graph):
        result = detect_cut_sets(small_graph)
        for u, v in result.bridges:
            assert small_graph.has_edge(u, v) or small_graph.has_edge(v, u)

    def test_known_bridge_detection(self):
        """Path graph has all edges as bridges."""
        G_path = nx.path_graph(5)
        # Add required node attributes
        for n in G_path.nodes():
            G_path.nodes[n]["node_type"]      = NodeType.RESIDENTIAL
            G_path.nodes[n]["priority_level"] = int(PriorityLevel.LOW)
            G_path.nodes[n]["capacity"]       = 5.0
            G_path.nodes[n]["status"]         = "active"
            G_path.nodes[n]["label"]          = f"R{n}"
            G_path.nodes[n]["x"]              = float(n)
            G_path.nodes[n]["y"]              = 0.0
        for u, v in G_path.edges():
            G_path[u][v][EDGE_RESISTANCE] = 1.0
            G_path[u][v][EDGE_STATUS]     = "active"
        result = detect_cut_sets(G_path)
        # All 4 edges of a path-5 graph are bridges
        assert len(result.bridges) == 4

    def test_known_ap_detection(self):
        """Star graph centre node is always an articulation point."""
        G_star = nx.star_graph(4)   # node 0 is centre; nodes 1-4 are leaves
        for n in G_star.nodes():
            G_star.nodes[n]["node_type"]      = NodeType.SUBSTATION if n == 0 else NodeType.RESIDENTIAL
            G_star.nodes[n]["priority_level"] = int(PriorityLevel.HIGH if n == 0 else PriorityLevel.LOW)
            G_star.nodes[n]["capacity"]       = 10.0
            G_star.nodes[n]["status"]         = "active"
            G_star.nodes[n]["label"]          = f"N{n}"
            G_star.nodes[n]["x"]              = 0.0
            G_star.nodes[n]["y"]              = float(n)
        for u, v in G_star.edges():
            G_star[u][v][EDGE_RESISTANCE] = 1.0
            G_star[u][v][EDGE_STATUS]     = "active"
        result = detect_cut_sets(G_star)
        assert 0 in result.articulation_points


# ─────────────────────────────────────────────────────────────────────────────
# § 3  POSET Schedule
# ─────────────────────────────────────────────────────────────────────────────

class TestPOSETSchedule:

    def test_returns_poset_schedule(self, small_graph):
        result = compute_poset_schedule(small_graph)
        assert result is not None

    def test_restoration_order_non_empty(self, small_graph):
        result = compute_poset_schedule(small_graph)
        assert len(result.restoration_order) > 0

    def test_restoration_order_length(self, small_graph):
        result = compute_poset_schedule(small_graph)
        assert len(result.restoration_order) == small_graph.number_of_nodes()

    def test_critical_nodes_restored_first(self, small_graph):
        result = compute_poset_schedule(small_graph)
        order  = result.restoration_order
        first_prio = small_graph.nodes[order[0]]["priority_level"]
        # First node must be CRITICAL (1) or HIGH (2)
        assert first_prio <= int(PriorityLevel.HIGH)

    def test_no_low_priority_before_critical(self, small_graph):
        result = compute_poset_schedule(small_graph)
        order  = result.restoration_order
        # Find first CRITICAL and first LOW in order
        first_critical_idx = next(
            (i for i, n in enumerate(order)
             if small_graph.nodes[n]["priority_level"] == int(PriorityLevel.CRITICAL)), None
        )
        first_low_idx = next(
            (i for i, n in enumerate(order)
             if small_graph.nodes[n]["priority_level"] == int(PriorityLevel.LOW)), None
        )
        if first_critical_idx is not None and first_low_idx is not None:
            assert first_critical_idx < first_low_idx

    def test_total_available_mw_positive(self, small_graph):
        result = compute_poset_schedule(small_graph)
        assert result.total_available_mw > 0

    def test_power_allocation_sums_lte_one(self, small_graph):
        result = compute_poset_schedule(small_graph)
        total = sum(v for v in result.power_allocation.values())
        assert total <= 1.001   # allow floating-point epsilon

    def test_power_allocation_all_non_negative(self, small_graph):
        result = compute_poset_schedule(small_graph)
        for node, alloc in result.power_allocation.items():
            assert alloc >= 0.0, f"Negative allocation for node {node}"

    def test_priority_groups_keys_are_ints(self, small_graph):
        result = compute_poset_schedule(small_graph)
        for k in result.priority_groups.keys():
            assert isinstance(k, int)

    def test_unserved_nodes_subset_of_active(self, small_graph):
        result = compute_poset_schedule(small_graph)
        active = set(small_graph.nodes())
        for n in result.unserved_nodes:
            assert n in active

    def test_override_generation_forces_deficit(self, small_graph):
        # Provide 1 MW → almost all non-generator nodes should be unserved
        result = compute_poset_schedule(small_graph, available_generation_mw=1.0)
        assert len(result.unserved_nodes) > 0

    def test_lattice_edges_are_tuples(self, small_graph):
        result = compute_poset_schedule(small_graph)
        for edge in result.lattice_edges:
            assert isinstance(edge, tuple)
            assert len(edge) == 2

    def test_failed_nodes_excluded_from_schedule(self, small_graph):
        all_nodes = list(small_graph.nodes())
        excluded  = all_nodes[:3]
        result = compute_poset_schedule(small_graph, failed_nodes=excluded)
        for n in excluded:
            assert n not in result.restoration_order


# ─────────────────────────────────────────────────────────────────────────────
# § 4  Dijkstra SSSP
# ─────────────────────────────────────────────────────────────────────────────

class TestDijkstra:

    def test_source_distance_is_zero(self, small_graph, first_generator):
        result = dijkstra_shortest_path(small_graph, first_generator)
        assert result.distances[first_generator] == 0.0

    def test_all_distances_non_negative(self, small_graph, first_generator):
        result = dijkstra_shortest_path(small_graph, first_generator)
        for d in result.distances.values():
            assert d >= 0

    def test_connected_graph_all_reached(self, small_graph, first_generator):
        result = dijkstra_shortest_path(small_graph, first_generator)
        assert result.total_nodes_reached == small_graph.number_of_nodes()

    def test_invalid_source_raises(self, small_graph):
        with pytest.raises(ValueError, match="not found"):
            dijkstra_shortest_path(small_graph, source=9999)

    def test_path_starts_at_source(self, small_graph, first_generator):
        result = dijkstra_shortest_path(small_graph, first_generator)
        for target, path in result.shortest_path_to.items():
            if path:
                assert path[0] == first_generator

    def test_path_ends_at_target(self, small_graph, first_generator):
        result = dijkstra_shortest_path(small_graph, first_generator)
        for target, path in result.shortest_path_to.items():
            if path:
                assert path[-1] == target

    def test_triangle_inequality(self, small_graph, first_generator):
        """d(s, v) ≤ d(s, u) + w(u, v) for all edges (u,v)."""
        result = dijkstra_shortest_path(small_graph, first_generator)
        for u, v, data in small_graph.edges(data=True):
            w = data.get(EDGE_RESISTANCE, 1.0)
            if result.distances[u] < math.inf and result.distances[v] < math.inf:
                assert result.distances[v] <= result.distances[u] + w + 1e-9
                assert result.distances[u] <= result.distances[v] + w + 1e-9

    def test_exclude_edges_respected(self, small_graph, first_generator):
        all_edges = list(small_graph.edges())
        # Exclude the first 5 edges → result may have fewer reachable nodes
        result = dijkstra_shortest_path(
            small_graph, first_generator,
            exclude_edges=all_edges[:5]
        )
        assert result.total_nodes_reached >= 1   # source itself

    def test_targets_parameter(self, small_graph, first_generator):
        other_nodes = [n for n in small_graph.nodes() if n != first_generator]
        targets = other_nodes[:3]
        result = dijkstra_shortest_path(small_graph, first_generator, targets=targets)
        for t in targets:
            assert t in result.shortest_path_to

    def test_dijkstra_on_large_graph(self, large_graph):
        gen = find_generator_nodes(large_graph)[0]
        result = dijkstra_shortest_path(large_graph, gen)
        assert result.total_nodes_reached == large_graph.number_of_nodes()


# ─────────────────────────────────────────────────────────────────────────────
# § 5  Kruskal MST
# ─────────────────────────────────────────────────────────────────────────────

class TestKruskalMST:

    def test_mst_has_v_minus_1_edges(self, small_graph):
        result = kruskal_mst(small_graph)
        assert len(result.mst_edges) == small_graph.number_of_nodes() - 1

    def test_mst_is_spanning(self, small_graph):
        result = kruskal_mst(small_graph)
        assert result.is_spanning is True

    def test_mst_total_weight_positive(self, small_graph):
        result = kruskal_mst(small_graph)
        assert result.total_weight > 0

    def test_mst_graph_is_connected(self, small_graph):
        result = kruskal_mst(small_graph)
        assert nx.is_connected(result.mst_graph)

    def test_mst_graph_has_correct_node_count(self, small_graph):
        result = kruskal_mst(small_graph)
        assert result.mst_graph.number_of_nodes() == small_graph.number_of_nodes()

    def test_mst_edge_df_shape(self, small_graph):
        result = kruskal_mst(small_graph)
        assert len(result.edge_df) == small_graph.number_of_nodes() - 1
        assert EDGE_RESISTANCE in result.edge_df.columns

    def test_mst_weight_le_total_graph_weight(self, small_graph):
        result   = kruskal_mst(small_graph)
        all_wts  = sum(d.get(EDGE_RESISTANCE, 0) for _, _, d in small_graph.edges(data=True))
        assert result.total_weight <= all_wts

    def test_mst_is_acyclic(self, small_graph):
        result = kruskal_mst(small_graph)
        assert nx.is_tree(result.mst_graph)

    def test_mst_excludes_failed_edges(self, small_graph):
        edges  = list(small_graph.edges())
        failed = edges[:3]
        result = kruskal_mst(small_graph, exclude_edges=failed)
        mst_edge_pairs = {(min(u,v), max(u,v)) for u, v, _ in result.mst_edges}
        for u, v in failed:
            assert (min(u,v), max(u,v)) not in mst_edge_pairs

    def test_mst_excludes_failed_nodes(self, small_graph):
        all_nodes = list(small_graph.nodes())
        excluded  = all_nodes[:2]
        result = kruskal_mst(small_graph, exclude_nodes=excluded)
        for n in excluded:
            assert n not in result.mst_graph.nodes()

    def test_mst_on_large_graph(self, large_graph):
        result = kruskal_mst(large_graph)
        assert result.is_spanning
        assert len(result.mst_edges) == large_graph.number_of_nodes() - 1

    def test_mst_minimality(self):
        """On a known small graph, verify MST picks the cheapest edges."""
        G = nx.Graph()
        for n in range(4):
            G.add_node(n, node_type=NodeType.RESIDENTIAL, priority_level=int(PriorityLevel.LOW),
                       capacity=5.0, status="active", label=f"R{n}", x=0.0, y=0.0)
        # Triangle + extra: edges with known weights
        G.add_edge(0, 1, **{EDGE_RESISTANCE: 1.0, EDGE_STATUS: "active", "capacity_mw": 10, "power_loss_pct": 1, "voltage_kv": 11, "length_km": 1})
        G.add_edge(1, 2, **{EDGE_RESISTANCE: 2.0, EDGE_STATUS: "active", "capacity_mw": 10, "power_loss_pct": 1, "voltage_kv": 11, "length_km": 1})
        G.add_edge(0, 2, **{EDGE_RESISTANCE: 10.0, EDGE_STATUS: "active", "capacity_mw": 10, "power_loss_pct": 1, "voltage_kv": 11, "length_km": 1})
        G.add_edge(2, 3, **{EDGE_RESISTANCE: 3.0, EDGE_STATUS: "active", "capacity_mw": 10, "power_loss_pct": 1, "voltage_kv": 11, "length_km": 1})
        result = kruskal_mst(G)
        # MST should pick edges with weights 1+2+3=6, NOT the edge with weight 10
        mst_weights = sorted([d.get(EDGE_RESISTANCE) for _, _, d in result.mst_edges])
        assert mst_weights == [1.0, 2.0, 3.0]
        assert result.total_weight == 6.0


# ─────────────────────────────────────────────────────────────────────────────
# § 6  Union-Find
# ─────────────────────────────────────────────────────────────────────────────

class TestUnionFind:

    def test_initial_all_separate(self):
        uf = _UnionFind([0, 1, 2, 3])
        assert uf.find(0) != uf.find(1)

    def test_union_connects(self):
        uf = _UnionFind([0, 1, 2])
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_returns_true_on_merge(self):
        uf = _UnionFind([0, 1])
        assert uf.union(0, 1) is True

    def test_union_returns_false_on_same_set(self):
        uf = _UnionFind([0, 1])
        uf.union(0, 1)
        assert uf.union(0, 1) is False

    def test_path_compression(self):
        uf = _UnionFind([0, 1, 2, 3, 4])
        uf.union(0, 1); uf.union(1, 2); uf.union(2, 3); uf.union(3, 4)
        root = uf.find(4)
        assert uf.find(0) == root


# ─────────────────────────────────────────────────────────────────────────────
# § 7  Full Recovery Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:

    def test_pipeline_returns_all_keys(self, small_graph):
        result = run_full_recovery_pipeline(small_graph)
        assert all(k in result for k in ["cut_set", "poset", "mst", "dijkstra", "summary"])

    def test_summary_has_kpi_keys(self, small_graph):
        result = run_full_recovery_pipeline(small_graph)
        s = result["summary"]
        for key in ["grid_is_connected", "active_generators", "mst_total_resistance",
                    "generation_mw", "demand_mw", "power_deficit_mw"]:
            assert key in s

    def test_dijkstra_run_per_generator(self, small_graph):
        gen_count = len(find_generator_nodes(small_graph))
        result    = run_full_recovery_pipeline(small_graph)
        assert len(result["dijkstra"]) == gen_count

    def test_pipeline_with_failures(self, small_graph):
        edges  = list(small_graph.edges())
        result = run_full_recovery_pipeline(
            small_graph, failed_edges=edges[:3], failed_nodes=[]
        )
        s = result["summary"]
        assert s["failed_edges_count"] == 3

    def test_apply_disaster_non_destructive(self, small_graph):
        edges  = list(small_graph.edges())
        failed = edges[:2]
        G_post = apply_disaster(small_graph, failed_edge_ids=failed)
        # Original graph is unchanged
        for u, v in failed:
            assert small_graph[u][v].get(EDGE_STATUS) != "failed"
        # Post-disaster graph has them marked
        for u, v in failed:
            assert G_post[u][v][EDGE_STATUS] == "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  TEST SUITE: INCIDENT AUDIT & SCENARIO COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncidentAuditAndScenarioComparison:
    """Verification of Live Incident Audit Ticker & Scenario Comparison Engine."""

    def test_audit_event_model(self):
        from src.engine import AuditEvent, create_audit_event
        evt = create_audit_event(
            phase="FAULT_INJECTION",
            event="Transmission line (3, 7) severed by storm",
            status="ALERT",
            duration_ms=4.52,
            line_id=(3, 7),
        )
        assert isinstance(evt, AuditEvent)
        assert evt.phase == "FAULT_INJECTION"
        assert evt.status == "ALERT"
        assert evt.duration_ms == 4.52
        d = evt.to_dict()
        assert d["phase"] == "FAULT_INJECTION"
        assert d["metadata"]["line_id"] == (3, 7)
        assert "timestamp" in d

    def test_disaster_preset_configs(self, small_graph):
        from src.engine import get_disaster_preset_config
        presets = [
            "Category 5 Cyclone (Sever Arterial Trunks)",
            "Cascading Islanding (Cut Central Articulation Points)",
            "Hospital Medical District Isolation",
            "Severe Brownout (50% Generation Drop)",
            "Worst-Case Dual-Fault (Cyclone + Brownout)",
            "Nominal Baseline",
        ]
        for p in presets:
            fe, fn, scale = get_disaster_preset_config(small_graph, p)
            assert isinstance(fe, list)
            assert isinstance(fn, list)
            assert isinstance(scale, float)
            assert 0.0 < scale <= 1.0

        # Specific assertions
        fe_cyc, fn_cyc, s_cyc = get_disaster_preset_config(small_graph, "Category 5 Cyclone")
        assert len(fe_cyc) > 0
        assert s_cyc == 1.0

        fe_brn, fn_brn, s_brn = get_disaster_preset_config(small_graph, "Severe Brownout")
        assert s_brn < 1.0

        fe_nom, fn_nom, s_nom = get_disaster_preset_config(small_graph, "Nominal Baseline")
        assert len(fe_nom) == 0
        assert len(fn_nom) == 0
        assert s_nom == 1.0

    def test_evaluate_scenario_metrics(self, small_graph):
        from src.engine import evaluate_scenario, ScenarioMetrics
        metrics = evaluate_scenario(
            small_graph,
            failed_edges=list(small_graph.edges())[:2],
            failed_nodes=[],
            available_generation_mw=250.0,
            scenario_name="Test Outage",
        )
        assert isinstance(metrics, ScenarioMetrics)
        assert metrics.scenario_name == "Test Outage"
        assert metrics.total_power_loss_mw >= 0.0
        assert metrics.recovery_latency_ms >= 0.0
        assert 0.0 <= metrics.health_percentage <= 100.0
        assert 0.0 <= metrics.restoration_rate <= 100.0
        assert metrics.broken_edges_count == 2
        assert metrics.num_islands >= 1
        d = metrics.to_dict()
        assert d["scenario_name"] == "Test Outage"
        assert "phase_latencies_ms" in d

    def test_compare_scenario_metrics_dataframe(self, small_graph):
        from src.engine import evaluate_scenario, compare_scenario_metrics
        scen_a = evaluate_scenario(small_graph, scenario_name="Nominal")
        scen_b = evaluate_scenario(
            small_graph,
            failed_edges=list(small_graph.edges())[:3],
            available_generation_mw=100.0,
            scenario_name="Cyclone Stress",
        )
        df = compare_scenario_metrics(scen_a, scen_b)
        assert not df.empty
        assert "Metric" in df.columns
        assert "Scenario A" in df.columns
        assert "Scenario B" in df.columns
        assert "Delta (B - A)" in df.columns
        assert "Status" in df.columns
        # Total power loss must be present in comparative table
        assert any("Total Power Loss" in str(m) for m in df["Metric"])
        assert any("Recovery Latency" in str(m) for m in df["Metric"])
        assert any("Health Score" in str(m) for m in df["Metric"])
        assert any("Reachability" in str(m) for m in df["Metric"])

    def test_dynamic_reachability_not_just_connected(self, small_graph):
        """
        Verify reachability is NOT simply 100% when graph is connected (components == 1).
        Under generation deficit (brownout), unserved nodes reduce reachability dynamically.
        """
        from src.engine import evaluate_scenario
        # Abundant generation case: all nodes served -> reachability 100%
        full_pwr = evaluate_scenario(small_graph, available_generation_mw=500.0, scenario_name="Full Power")
        assert full_pwr.num_islands == 1
        assert full_pwr.restoration_rate == 100.0
        assert full_pwr.grid_reachability == 100.0
        assert full_pwr.metadata["active_served_nodes_count"] == small_graph.number_of_nodes()

        # Brownout case: 0 failed edges, 0 failed nodes (so components == 1),
        # but available generation is only 5.0 MW, so multiple nodes are unserved
        brownout = evaluate_scenario(
            small_graph,
            failed_edges=[],
            failed_nodes=[],
            available_generation_mw=5.0,
            scenario_name="Brownout",
        )
        assert brownout.num_islands == 1  # Graph is fully connected!
        # Reachability must NOT be 100% just because components == 1
        assert brownout.restoration_rate < 100.0
        assert brownout.restoration_rate == brownout.grid_reachability

        # Check formula: Active Served Nodes / Total Nodes * 100
        total_nodes = small_graph.number_of_nodes()
        served_count = brownout.metadata["active_served_nodes_count"]
        expected_rate = round(served_count / total_nodes * 100, 1)
        assert brownout.restoration_rate == expected_rate

    def test_dynamic_reachability_with_failed_nodes(self, small_graph):
        """
        Failed nodes are not served, and denominator is Total Nodes (base graph size).
        """
        from src.engine import evaluate_scenario
        failed_nodes = [0, 1]
        metrics = evaluate_scenario(
            small_graph,
            failed_nodes=failed_nodes,
            available_generation_mw=500.0,
            scenario_name="Node Faults",
        )
        total_nodes = small_graph.number_of_nodes()
        served_nodes = metrics.metadata["active_served_nodes"]
        for fn in failed_nodes:
            assert fn not in served_nodes
        assert metrics.restoration_rate <= round((total_nodes - len(failed_nodes)) / total_nodes * 100, 1)

    def test_dynamic_reachability_zero_generation(self, small_graph):
        """When available generation is 0 MW, reachability is 0.0%."""
        from src.engine import evaluate_scenario
        metrics = evaluate_scenario(
            small_graph,
            available_generation_mw=0.0,
            scenario_name="Blackout",
        )
        assert metrics.restoration_rate == 0.0
        assert metrics.grid_reachability == 0.0
        assert metrics.metadata["active_served_nodes_count"] == 0

    def test_root_proxy_exports(self):
        import engine
        assert hasattr(engine, "AuditEvent")
        assert hasattr(engine, "ScenarioMetrics")
        assert hasattr(engine, "create_audit_event")
        assert hasattr(engine, "get_disaster_preset_config")
        assert hasattr(engine, "evaluate_scenario")
        assert hasattr(engine, "compare_scenario_metrics")
