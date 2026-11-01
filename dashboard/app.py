"""
dashboard/app.py — Smart Grid Disaster Recovery & Rerouting Optimization
=========================================================================
Main Streamlit entry-point.

Run with:
    streamlit run dashboard/app.py

Layout
------
  ┌─────────────────────────────────────────────────────────────────┐
  │  SIDEBAR  Control Center (topology, disaster, generator sliders)│
  ├─────────────────────────────────────────────────────────────────┤
  │  TOP METRICS ROW  (5 KPI cards)                                 │
  ├────────────────────────────┬────────────────────────────────────┤
  │  LEFT  Network Graph       │  RIGHT  Cut-Set & POSET tables     │
  └────────────────────────────┴────────────────────────────────────┘
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# ── resolve package root so `src` is importable regardless of CWD ──────────
_PROJ_ROOT = Path(__file__).parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from src.engine import (
    NodeType,
    PriorityLevel,
    apply_disaster,
    compute_poset_schedule,
    detect_cut_sets,
    get_node_dataframe,
    grid_metrics,
    initialize_grid_graph,
    run_full_recovery_pipeline,
)

# ═══════════════════════════════════════════════════════════════════════════
# §0  PAGE CONFIG — must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Grid | Disaster Recovery Control Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Smart Grid Disaster Recovery & Rerouting Optimization System**\n\n"
            "B.Tech CSE — Discrete Mathematics Project  \n"
            "Algorithms: Graph Theory · Cut-Set Analysis · POSET · Dijkstra · Kruskal MST"
        )
    },
)

# ═══════════════════════════════════════════════════════════════════════════
# §1  GLOBAL CSS — industrial dark control-centre aesthetic
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Google Font import ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root tokens ── */
    :root {
        --bg-primary:   #0a0e1a;
        --bg-card:      #111827;
        --bg-card-glow: #131d30;
        --border:       #1e2d45;
        --border-glow:  #1e4976;
        --accent-blue:  #3b82f6;
        --accent-cyan:  #06b6d4;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red:   #ef4444;
        --accent-purple:#8b5cf6;
        --text-primary: #f0f6ff;
        --text-muted:   #64748b;
        --text-dim:     #374151;
        --radius-card:  14px;
        --shadow-card:  0 4px 24px rgba(0,0,0,0.55);
        --transition:   0.22s cubic-bezier(.4,0,.2,1);
    }

    /* ── App shell ── */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    section[data-testid="stSidebar"] {
        background: #0d1525;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

    /* ── Hide default decoration ── */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding: 1.6rem 2rem 2rem; }

    /* ── KPI Metric cards ── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-card);
        padding: 1.1rem 1.3rem 1rem;
        text-align: center;
        transition: border-color var(--transition), box-shadow var(--transition);
        box-shadow: var(--shadow-card);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        border-radius: var(--radius-card) var(--radius-card) 0 0;
    }
    .kpi-card:hover {
        border-color: var(--border-glow);
        box-shadow: 0 0 22px rgba(59,130,246,0.18), var(--shadow-card);
    }
    .kpi-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .kpi-value.green  { background: linear-gradient(135deg,#10b981,#34d399); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
    .kpi-value.red    { background: linear-gradient(135deg,#ef4444,#f87171); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
    .kpi-value.amber  { background: linear-gradient(135deg,#f59e0b,#fbbf24); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
    .kpi-value.purple { background: linear-gradient(135deg,#8b5cf6,#a78bfa); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
    .kpi-sub {
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Section headers ── */
    .section-header {
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.75rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-muted); margin: 0.2rem 0 0.9rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid var(--border);
    }
    .section-dot {
        width:8px; height:8px; border-radius:50%;
        background: var(--accent-cyan);
        box-shadow: 0 0 8px var(--accent-cyan);
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 0.18em 0.7em;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-online  { background:rgba(16,185,129,.15); color:#34d399; border:1px solid rgba(16,185,129,.35); }
    .badge-fault   { background:rgba(239,68,68,.15);  color:#f87171; border:1px solid rgba(239,68,68,.35); }
    .badge-warn    { background:rgba(245,158,11,.15); color:#fbbf24; border:1px solid rgba(245,158,11,.35); }
    .badge-critical{ background:rgba(139,92,246,.15); color:#a78bfa; border:1px solid rgba(139,92,246,.35); }

    /* ── Sidebar controls polish ── */
    .sidebar-group-title {
        font-size: 0.65rem; font-weight: 700;
        letter-spacing: 0.14em; text-transform: uppercase;
        color: #4b6a8f; margin: 1.2rem 0 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #1a2540;
    }

    /* ── Streamlit element overrides ── */
    .stSlider [data-testid="stThumbValue"] { font-family:'JetBrains Mono',monospace; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    .stMultiSelect > div { border-radius: 8px; }

    /* ── App title bar ── */
    .app-title-bar {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 1.4rem;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid var(--border);
    }
    .app-title {
        font-size: 1.45rem; font-weight: 800;
        background: linear-gradient(90deg,#e0f2fe,#7dd3fc,#3b82f6);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip: text; letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.72rem; color: var(--text-muted);
        letter-spacing: 0.06em; margin-top: 0.15rem;
    }
    .live-badge {
        display:inline-flex; align-items:center; gap:0.4rem;
        background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.3);
        color:#34d399; border-radius:999px; padding:0.3rem 0.85rem;
        font-size:0.68rem; font-weight:700; letter-spacing:0.1em;
    }
    .live-dot {
        width:7px; height:7px; border-radius:50%;
        background:#10b981; animation: pulse 1.4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%,100%{ opacity:1; transform:scale(1); }
        50%{ opacity:.45; transform:scale(1.35); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# §2  HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _kpi_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    """Return HTML for a single KPI card."""
    cls = f"kpi-value {color}".strip()
    return (
        f'<div class="kpi-card">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="{cls}">{value}</div>'
        f'  <div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def _section_header(text: str) -> None:
    st.markdown(
        f'<div class="section-header"><span class="section-dot"></span>{text}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _cached_graph(num_nodes: int, seed: int = 42) -> nx.Graph:
    """Cache the base graph so topology changes do not re-run on every widget tweak."""
    return initialize_grid_graph(num_nodes=num_nodes, seed=seed)


def _parse_edge_key(s: str) -> tuple[int, int]:
    """Convert '3 — 7' back to (3, 7)."""
    parts = s.replace("\u2014", "-").replace("\u2013", "-").split("-")
    return int(parts[0].strip()), int(parts[1].strip())


# ═══════════════════════════════════════════════════════════════════════════
# §3  SIDEBAR — CONTROL CENTER
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Brand header ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;padding:0.5rem 0 1rem;">
            <div style="font-size:2rem;margin-bottom:0.2rem;">&#9889;</div>
            <div style="font-size:1rem;font-weight:800;letter-spacing:-0.01em;
                        background:linear-gradient(90deg,#7dd3fc,#3b82f6);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;">
                Grid Control Center
            </div>
            <div style="font-size:0.64rem;color:#4b6a8f;letter-spacing:0.1em;
                        text-transform:uppercase;margin-top:0.15rem;">
                Disaster Recovery System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── § 3.1  Topology Configuration ─────────────────────────────────────
    st.markdown('<div class="sidebar-group-title">Topology Configuration</div>', unsafe_allow_html=True)

    num_nodes: int = st.slider(
        "Substations / Nodes",
        min_value=10,
        max_value=50,
        value=20,
        step=2,
        help="Total number of substation nodes in the simulated power grid.",
    )

    seed_val: int = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Seed for reproducible graph generation.",
    )

    # Build the base (un-damaged) graph and derive selectable edges/nodes
    with st.spinner("Building grid topology..."):
        base_graph: nx.Graph = _cached_graph(num_nodes=num_nodes, seed=int(seed_val))

    all_edges   = sorted(base_graph.edges())
    all_nodes   = sorted(base_graph.nodes())
    edge_labels = [f"{u} - {v}" for u, v in all_edges]
    node_labels = [f"Node {n}" for n in all_nodes]

    gen_count = sum(
        1 for _, d in base_graph.nodes(data=True)
        if d.get("node_type") == NodeType.GENERATOR
    )
    st.caption(
        f"Edges: {base_graph.number_of_edges()} transmission lines  |  "
        f"Generators: {gen_count}"
    )

    st.divider()

    # ── § 3.2  Disaster Simulation Triggers ───────────────────────────────
    st.markdown('<div class="sidebar-group-title">Disaster Simulation Triggers</div>', unsafe_allow_html=True)

    failed_edge_labels: list[str] = st.multiselect(
        "Break Transmission Lines",
        options=edge_labels,
        default=[],
        placeholder="Select lines to sever...",
        help="Simulate downed transmission lines (e.g., flood, storm damage).",
    )

    failed_node_labels: list[str] = st.multiselect(
        "Disable Substation Nodes",
        options=node_labels,
        default=[],
        placeholder="Select nodes to disable...",
        help="Simulate substation failures (e.g., explosion, fire).",
    )

    # Convert selections back to ints/tuples
    failed_edges: list[tuple[int, int]] = [_parse_edge_key(s) for s in failed_edge_labels]
    failed_nodes: list[int]              = [int(s.replace("Node ", "")) for s in failed_node_labels]

    st.divider()

    # ── § 3.3  Generator Capacity Configuration ───────────────────────────
    st.markdown('<div class="sidebar-group-title">Generator Capacity (MW)</div>', unsafe_allow_html=True)

    generator_nodes = [
        n for n, d in base_graph.nodes(data=True)
        if d.get("node_type") == NodeType.GENERATOR
    ]

    gen_capacities: dict[int, float] = {}
    if generator_nodes:
        for gn in generator_nodes:
            default_cap = float(base_graph.nodes[gn].get("capacity_mw", 100.0))
            gen_capacities[gn] = st.slider(
                f"Generator G{gn}",
                min_value=10.0,
                max_value=500.0,
                value=min(default_cap, 500.0),
                step=10.0,
                format="%.0f MW",
                key=f"gen_slider_{gn}",
                help=f"Active output capacity of Generator node {gn}.",
            )
    else:
        st.info("No generator nodes detected in current topology.")

    total_gen_mw = sum(gen_capacities.values()) if gen_capacities else 200.0

    st.divider()

    # ── § 3.4  Run pipeline button ─────────────────────────────────────────
    run_btn = st.button(
        "Run Recovery Pipeline",
        use_container_width=True,
        type="primary",
        help="Execute: Cut-Set -> POSET -> Dijkstra -> Kruskal recovery analysis.",
    )

    st.markdown(
        '<div style="text-align:center;font-size:0.63rem;color:#2d4260;margin-top:1rem;">'
        'B.Tech CSE &middot; Discrete Mathematics &middot; 2026'
        '</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════
# §4  COMPUTE — apply disaster & run analysis
# ═══════════════════════════════════════════════════════════════════════════

t0 = time.perf_counter()

damaged_graph: nx.Graph = apply_disaster(
    base_graph,
    failed_edge_ids=failed_edges if failed_edges else None,
    failed_node_ids=failed_nodes if failed_nodes else None,
)

cutset_result = detect_cut_sets(
    damaged_graph,
    failed_edges=failed_edges if failed_edges else None,
    failed_nodes=failed_nodes if failed_nodes else None,
)

poset_schedule = compute_poset_schedule(
    damaged_graph,
    available_generation_mw=total_gen_mw,
    failed_nodes=failed_nodes if failed_nodes else None,
)

pipeline_result: dict[str, Any] = {}
if run_btn:
    pipeline_result = run_full_recovery_pipeline(
        base_graph,
        failed_edges=failed_edges if failed_edges else None,
        failed_nodes=failed_nodes if failed_nodes else None,
    )

latency_ms = (time.perf_counter() - t0) * 1000

# ── Derived metrics ────────────────────────────────────────────────────────
active_link_count = sum(
    1 for _, _, d in damaged_graph.edges(data=True)
    if d.get("status", "active") == "active"
)
broken_link_count = len(failed_edges)


def _restoration_rate(graph: nx.Graph, gen_nodes: list[int], failed: list[int]) -> float:
    """Fraction of live nodes reachable from at least one active generator."""
    if not gen_nodes:
        return 0.0
    reachable: set[int] = set()
    for gn in gen_nodes:
        if gn in graph and gn not in failed:
            reachable |= nx.node_connected_component(graph, gn)
    live_nodes = [n for n in graph.nodes() if n not in failed]
    if not live_nodes:
        return 0.0
    return round(len([n for n in live_nodes if n in reachable]) / len(live_nodes) * 100, 1)


restoration_rate = _restoration_rate(damaged_graph, generator_nodes, failed_nodes)

# ═══════════════════════════════════════════════════════════════════════════
# §5  TITLE BAR
# ═══════════════════════════════════════════════════════════════════════════

system_ok = not failed_edges and not failed_nodes
status_badge = (
    '<span class="badge badge-online">&#9679; NOMINAL</span>'
    if system_ok
    else '<span class="badge badge-fault">&#9888; FAULT ACTIVE</span>'
)

st.markdown(
    f"""
    <div class="app-title-bar">
        <div>
            <div class="app-title">&#9889; Smart Grid Disaster Recovery</div>
            <div class="app-subtitle">
                Discrete Mathematics &mdash; Graph Theory &middot; Cut-Set Analysis
                &middot; POSET Scheduling &middot; Dijkstra &middot; Kruskal MST
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:0.75rem;">
            {status_badge}
            <div class="live-badge">
                <span class="live-dot"></span>LIVE
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════
# §6  TOP METRICS ROW — 5 KPI cards
# ═══════════════════════════════════════════════════════════════════════════

_section_header("SYSTEM KEY PERFORMANCE INDICATORS")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        _kpi_card("Total Nodes", str(damaged_graph.number_of_nodes()),
                  sub=f"{len(generator_nodes)} generators"),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        _kpi_card("Active Links", str(active_link_count),
                  sub=f"of {base_graph.number_of_edges()} total", color="green"),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        _kpi_card("Broken Links", str(broken_link_count),
                  sub=f"{len(failed_nodes)} nodes offline",
                  color="red" if broken_link_count > 0 else "green"),
        unsafe_allow_html=True,
    )
with c4:
    rate_color = "green" if restoration_rate >= 80 else ("amber" if restoration_rate >= 50 else "red")
    st.markdown(
        _kpi_card("Restoration Rate", f"{restoration_rate}%",
                  sub="grid reachability", color=rate_color),
        unsafe_allow_html=True,
    )
with c5:
    lat_color = "green" if latency_ms < 200 else "amber"
    st.markdown(
        _kpi_card("Fault Detection Latency", f"{latency_ms:.1f} ms",
                  sub="< 200 ms target", color=lat_color),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# §7  SPLIT-SCREEN — LEFT: Network Graph  |  RIGHT: Analysis Tables
# ═══════════════════════════════════════════════════════════════════════════

left_col, right_col = st.columns([1.15, 1], gap="large")

# ────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Interactive Network Graph (NetworkX + Matplotlib)
# ────────────────────────────────────────────────────────────────────────────
with left_col:
    _section_header("GRID TOPOLOGY — NETWORK GRAPH")

    pos = nx.spring_layout(base_graph, seed=int(seed_val), k=1.8 / (num_nodes ** 0.45))

    failed_edge_set = set(frozenset(e) for e in failed_edges)
    failed_node_set = set(failed_nodes)
    art_points      = set(cutset_result.articulation_points)
    bridge_set      = set(frozenset(b) for b in cutset_result.bridges)

    # Node colours & sizes
    node_colors: list[str] = []
    node_sizes: list[int] = []
    for n in base_graph.nodes():
        ntype = base_graph.nodes[n].get("node_type", NodeType.SUBSTATION)
        if n in failed_node_set:
            node_colors.append("#374151"); node_sizes.append(160)
        elif ntype == NodeType.GENERATOR:
            node_colors.append("#f59e0b"); node_sizes.append(300)
        elif ntype in (NodeType.HOSPITAL, NodeType.EMERGENCY):
            node_colors.append("#3b82f6"); node_sizes.append(250)
        elif n in art_points:
            node_colors.append("#8b5cf6"); node_sizes.append(230)
        else:
            node_colors.append("#06b6d4"); node_sizes.append(175)

    # Edge classification
    solid_edges, dashed_edges = [], []
    solid_colors, dashed_colors = [], []
    solid_widths, dashed_widths = [], []

    for u, v in base_graph.edges():
        eset = frozenset([u, v])
        if eset in failed_edge_set:
            dashed_edges.append((u, v)); dashed_colors.append("#ef4444"); dashed_widths.append(2.2)
        elif eset in bridge_set:
            solid_edges.append((u, v)); solid_colors.append("#f59e0b"); solid_widths.append(1.9)
        else:
            solid_edges.append((u, v)); solid_colors.append("#10b981"); solid_widths.append(1.2)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.patch.set_facecolor("#0d1220")
    ax.set_facecolor("#0d1220")

    if solid_edges:
        nx.draw_networkx_edges(base_graph, pos, edgelist=solid_edges,
                               edge_color=solid_colors, width=solid_widths,
                               ax=ax, alpha=0.85)
    if dashed_edges:
        nx.draw_networkx_edges(base_graph, pos, edgelist=dashed_edges,
                               edge_color=dashed_colors, width=dashed_widths,
                               ax=ax, alpha=0.9, style="dashed")

    nx.draw_networkx_nodes(base_graph, pos,
                           node_color=node_colors, node_size=node_sizes,
                           ax=ax, alpha=0.95, linewidths=1.2, edgecolors="#1e2d45")

    nx.draw_networkx_labels(base_graph, pos, ax=ax,
                            font_size=7, font_color="#e2e8f0", font_family="monospace")

    legend_handles = [
        mpatches.Patch(color="#10b981", label="Active Line"),
        mpatches.Patch(color="#ef4444", label="Severed Line (Disaster)"),
        mpatches.Patch(color="#f59e0b", label="Critical Bridge / Generator"),
        mpatches.Patch(color="#3b82f6", label="Hospital / Priority Hub"),
        mpatches.Patch(color="#8b5cf6", label="Articulation Point"),
        mpatches.Patch(color="#374151", label="Offline Node"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
              framealpha=0.28, facecolor="#0a0e1a", edgecolor="#1e2d45",
              labelcolor="#94a3b8")

    ax.set_title(
        f"Grid Topology  |  {num_nodes} nodes  |  {base_graph.number_of_edges()} edges",
        color="#64748b", fontsize=9, pad=10, fontfamily="monospace",
    )
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Node Attribute Table", expanded=False):
        node_df = get_node_dataframe(base_graph)
        st.dataframe(node_df, use_container_width=True, height=250)

# ────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Cut-Set Isolation & POSET Priority Tables
# ────────────────────────────────────────────────────────────────────────────
with right_col:
    _section_header("CUT-SET ISOLATION BREAKDOWN")

    # Build edge status table
    rows = []
    for u, v in base_graph.edges():
        eset = frozenset([u, v])
        if eset in bridge_set or eset in failed_edge_set:
            status = "SEVERED" if eset in failed_edge_set else "BRIDGE"
            rows.append({
                "From": u, "To": v,
                "Resistance (Ohm)": base_graph.edges[u, v].get("resistance", "—"),
                "Capacity (MW)":    base_graph.edges[u, v].get("capacity_mw", "—"),
                "Status": status,
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=220)
    else:
        st.info("No cut-sets detected — grid is fully resilient under current configuration.")

    if cutset_result.articulation_points:
        st.caption(f"Articulation Points Detected: {len(cutset_result.articulation_points)}")
        ap_rows = [
            {
                "Node ID":  n,
                "Type":     base_graph.nodes[n].get("node_type", "Substation"),
                "Priority": PriorityLevel(
                    base_graph.nodes[n].get("priority_level", 4)
                ).name,
            }
            for n in cutset_result.articulation_points
        ]
        st.dataframe(pd.DataFrame(ap_rows), use_container_width=True, height=160)
    else:
        st.caption("No articulation points — no single-node failure disconnects the grid.")

    # Blacked-out zones
    blacked = cutset_result.blacked_out_zones
    zone_rows = [
        {
            "Priority Level": (plevel.name if hasattr(plevel, "name") else str(plevel)),
            "Affected Nodes": len(nodes),
            "Node IDs": ", ".join(map(str, nodes[:8])) + ("..." if len(nodes) > 8 else ""),
        }
        for plevel, nodes in blacked.items() if nodes
    ]
    if zone_rows:
        st.caption("Blacked-Out Zones by Priority")
        st.dataframe(pd.DataFrame(zone_rows), use_container_width=True, height=130)

    st.markdown("<br>", unsafe_allow_html=True)
    _section_header("POSET PRIORITY ALLOCATION")

    priority_name_map = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW", 5: "OFFLINE"}
    poset_rows = []
    for rank, node_id in enumerate(poset_schedule.restoration_order, start=1):
        p_int = base_graph.nodes[node_id].get("priority_level", 4)
        alloc = poset_schedule.power_allocation.get(node_id, 0.0)
        poset_rows.append({
            "Rank":       rank,
            "Node ID":    node_id,
            "Type":       base_graph.nodes[node_id].get("node_type", "Substation"),
            "Priority":   priority_name_map.get(p_int, "UNKNOWN"),
            "Alloc (MW)": f"{alloc * poset_schedule.total_available_mw:.1f}",
            "Served":     "Yes" if node_id not in poset_schedule.unserved_nodes else "No",
        })

    if poset_rows:
        st.dataframe(pd.DataFrame(poset_rows), use_container_width=True, height=330)
        served   = len(poset_schedule.restoration_order) - len(poset_schedule.unserved_nodes)
        unserved = len(poset_schedule.unserved_nodes)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Nodes Served",  served)
        mc2.metric("Unserved",       unserved)
        mc3.metric("Total Gen (MW)", f"{poset_schedule.total_available_mw:.0f}")
    else:
        st.info("No POSET schedule generated. Adjust generator sliders and refresh.")

# ═══════════════════════════════════════════════════════════════════════════
# §8  PIPELINE RESULTS (shown only after Run button)
# ═══════════════════════════════════════════════════════════════════════════

if pipeline_result:
    st.divider()
    _section_header("FULL RECOVERY PIPELINE RESULTS")

    r_left, r_right = st.columns(2, gap="large")

    with r_left:
        st.subheader("Dijkstra Rerouting Paths")
        reroute = pipeline_result.get("rerouting_paths", {})
        if reroute:
            for key, path_info in list(reroute.items())[:8]:
                cost = path_info.get("cost", "—")
                path = path_info.get("path", [])
                path_str = " -> ".join(map(str, path)) if path else "No path"
                st.markdown(
                    f'<div style="font-family:monospace;font-size:0.75rem;'
                    f'padding:0.4rem 0.65rem;margin:0.2rem 0;'
                    f'background:#111827;border-radius:7px;'
                    f'border-left:3px solid #3b82f6;">'
                    f'<span style="color:#60a5fa;font-weight:700;">{key}</span>'
                    f'<span style="color:#475569;"> &rarr; </span>'
                    f'<span style="color:#e2e8f0;">{path_str}</span>'
                    f'<span style="color:#475569;float:right;">{cost} Ohm</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No rerouting paths computed (no failed edges selected).")

    with r_right:
        st.subheader("Kruskal MST Recovery Backbone")
        mst_edges = pipeline_result.get("mst_edges", [])
        if mst_edges:
            mst_df = pd.DataFrame(
                [(u, v, w) for u, v, w in mst_edges],
                columns=["From", "To", "Weight (Ohm)"],
            )
            st.dataframe(mst_df, use_container_width=True, height=260)
            total_w = sum(w for _, _, w in mst_edges)
            st.caption(f"MST total resistance: {total_w:.3f} Ohm  |  {len(mst_edges)} backbone edges")
        else:
            st.info("No MST data. Run the recovery pipeline.")

# ═══════════════════════════════════════════════════════════════════════════
# §9  FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="border-top:1px solid #1e2d45;padding-top:0.9rem;margin-top:0.5rem;
                display:flex;justify-content:space-between;align-items:center;
                font-size:0.65rem;color:#2d4260;letter-spacing:0.06em;">
        <span>&#9889; Smart Grid Disaster Recovery &amp; Rerouting Optimization System</span>
        <span>B.Tech CSE &middot; Discrete Mathematics &middot; 2026
              &middot; Graph Theory &middot; Cut-Set &middot; POSET &middot; Dijkstra &middot; Kruskal</span>
    </div>
    """,
    unsafe_allow_html=True,
)
