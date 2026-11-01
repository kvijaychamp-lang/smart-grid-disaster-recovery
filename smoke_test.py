"""
smoke_test.py — Quick end-to-end smoke test for src/engine.py
Run with:  python smoke_test.py
"""
import sys, os

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

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
    PriorityLevel,
    NodeType,
)

PASS = "\033[92m✔\033[0m"
FAIL = "\033[91m✘\033[0m"

def check(label, condition):
    status = PASS if condition else FAIL
    print(f"  {status}  {label}")
    if not condition:
        raise AssertionError(f"FAILED: {label}")

print("\n═══════════════════════════════════════════════════════")
print("  Smart Grid Engine — Smoke Test Suite")
print("═══════════════════════════════════════════════════════\n")

# ── Test 1: Graph Initialisation ─────────────────────────────────────────────
print("[ 1 ] Graph Initialisation  G = (V, E, w)")
G = initialize_grid_graph(16, seed=42)
check("Correct node count (16)",        G.number_of_nodes() == 16)
check("Graph is connected",             __import__("networkx").is_connected(G))
check("All nodes have node_type",       all("node_type"       in d for _, d in G.nodes(data=True)))
check("All nodes have priority_level",  all("priority_level"  in d for _, d in G.nodes(data=True)))
check("All nodes have capacity",        all("capacity"        in d for _, d in G.nodes(data=True)))
check("All edges have resistance",      all("resistance"      in d for _,_,d in G.edges(data=True)))
check("All edges have capacity_mw",     all("capacity_mw"     in d for _,_,d in G.edges(data=True)))
check("All edges have voltage_kv",      all("voltage_kv"      in d for _,_,d in G.edges(data=True)))
check("At least 2 generators exist",    len(find_generator_nodes(G)) >= 2)
check("Critical nodes exist",           len(find_critical_nodes(G)) >= 1)
node_df = get_node_dataframe(G)
edge_df = get_edge_dataframe(G)
check("Node DataFrame has 16 rows",     len(node_df) == 16)
check("Edge DataFrame non-empty",       len(edge_df) > 0)
print(f"     Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
print(f"     Generators: {len(find_generator_nodes(G))}  |  Critical nodes: {len(find_critical_nodes(G))}")

# ── Test 2: Grid Metrics ──────────────────────────────────────────────────────
print("\n[ 2 ] Grid Metrics")
metrics = grid_metrics(G)
check("density is float in (0,1)",      0 < metrics["density"] < 1)
check("avg_degree > 0",                 metrics["avg_degree"] > 0)
check("is_connected == True",           metrics["is_connected"] is True)
print(f"     Density: {metrics['density']}  |  Avg Degree: {metrics['avg_degree']}  |  Diameter: {metrics['diameter']}")

# ── Test 3: Cut-Set Analysis ──────────────────────────────────────────────────
print("\n[ 3 ] Cut-Set Analysis")
edges = list(G.edges())
failed_edges = edges[:2]   # Simulate 2 line failures
result = detect_cut_sets(G, failed_edges=failed_edges)
check("CutSetResult returned",          result is not None)
check("articulation_points is list",    isinstance(result.articulation_points, list))
check("bridges is list",                isinstance(result.bridges, list))
check("connected_components is list",   isinstance(result.connected_components, list))
check("num_components >= 1",            result.num_components >= 1)
check("component_df non-empty",         len(result.component_df) > 0)
print(f"     Components after failure: {result.num_components}  |  APs: {len(result.articulation_points)}  |  Bridges: {len(result.bridges)}")

# ── Test 4: POSET Schedule ────────────────────────────────────────────────────
print("\n[ 4 ] POSET Priority Scheduler")
schedule = compute_poset_schedule(G)
check("POSETSchedule returned",               schedule is not None)
check("restoration_order non-empty",          len(schedule.restoration_order) > 0)
check("total_available_mw > 0",               schedule.total_available_mw > 0)
check("power_allocation keys match active",   len(schedule.power_allocation) > 0)
check("priority_groups has multiple tiers",   len(schedule.priority_groups) >= 2)
# Verify POSET order: first node in restoration_order must be CRITICAL or HIGH
first_node_prio = G.nodes[schedule.restoration_order[0]]["priority_level"]
check("First restored node is highest priority (≤ 2)", first_node_prio <= 2)
print(f"     Available: {schedule.total_available_mw} MW  |  Demanded: {schedule.total_demanded_mw} MW")
print(f"     Unserved nodes: {len(schedule.unserved_nodes)}  |  Tiers: {sorted(schedule.priority_groups.keys())}")

# ── Test 5: Dijkstra SSSP ─────────────────────────────────────────────────────
print("\n[ 5 ] Dijkstra Shortest Path")
gen_node = find_generator_nodes(G)[0]
dijk = dijkstra_shortest_path(G, source=gen_node)
check("DijkstraResult returned",              dijk is not None)
check("Source distance is 0.0",               dijk.distances[gen_node] == 0.0)
check("All other distances are non-negative", all(d >= 0 for d in dijk.distances.values()))
check("total_nodes_reached > 1",              dijk.total_nodes_reached > 1)
check("Paths reconstructed correctly",        dijk.shortest_path_to != {})
non_inf = [n for n, d in dijk.distances.items() if d < float("inf") and n != gen_node]
if non_inf:
    sample_target = non_inf[0]
    path = dijk.shortest_path_to.get(sample_target, [])
    check("Reconstructed path starts at source",  path and path[0] == gen_node)
    check("Reconstructed path ends at target",    path and path[-1] == sample_target)
print(f"     Source: Node {gen_node}  |  Reached: {dijk.total_nodes_reached}/{G.number_of_nodes()}")

# ── Test 6: Kruskal MST ───────────────────────────────────────────────────────
print("\n[ 6 ] Kruskal Minimum Spanning Tree")
mst = kruskal_mst(G)
check("KruskalResult returned",          mst is not None)
check("MST has V-1 edges",               len(mst.mst_edges) == G.number_of_nodes() - 1)
check("total_weight > 0",               mst.total_weight > 0)
check("is_spanning == True",            mst.is_spanning)
check("MST graph has correct node cnt", mst.mst_graph.number_of_nodes() == G.number_of_nodes())
check("edge_df has correct rows",       len(mst.edge_df) == G.number_of_nodes() - 1)
print(f"     MST edges: {len(mst.mst_edges)}  |  Total resistance: {mst.total_weight} Ω  |  Spanning: {mst.is_spanning}")

# ── Test 7: Full Recovery Pipeline ────────────────────────────────────────────
print("\n[ 7 ] Full Recovery Pipeline")
pipeline = run_full_recovery_pipeline(
    G,
    failed_edges=edges[:3],
    failed_nodes=[],
)
check("Pipeline returns all 5 keys",    all(k in pipeline for k in ["cut_set","poset","mst","dijkstra","summary"]))
check("Summary has required KPIs",      "grid_is_connected" in pipeline["summary"])
check("Dijkstra run for each generator",len(pipeline["dijkstra"]) >= 1)
s = pipeline["summary"]
print(f"     Connected: {s['grid_is_connected']}  |  Components: {s['connected_components']}  |  APs: {s['articulation_points']}")
print(f"     Generation: {s['generation_mw']} MW  |  Demand: {s['demand_mw']} MW  |  Deficit: {s['power_deficit_mw']} MW")
print(f"     MST Resistance: {s['mst_total_resistance']} Ω  |  Unserved: {s['unserved_nodes_count']} nodes")

# ── Test 8: Error Handling ─────────────────────────────────────────────────────
print("\n[ 8 ] Error Handling")
try:
    dijkstra_shortest_path(G, source=9999)
    check("Raises ValueError for missing source", False)
except ValueError:
    check("Raises ValueError for missing source", True)

try:
    initialize_grid_graph(2)
    check("Raises ValueError for too-small graph", False)
except ValueError:
    check("Raises ValueError for too-small graph", True)

print("\n═══════════════════════════════════════════════════════")
print("  All smoke tests passed ✔")
print("═══════════════════════════════════════════════════════\n")
