"""Dependency Graph Visualization - Interactive formula dependency explorer.

This component displays the relationships between calculations, showing
how values flow through the model from inputs to outputs.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Dict, List, Set, Tuple
import math

from src.calculations.formula_registry import FormulaRegistry, FormulaCategory, FormulaDefinition


def _get_category_color(category: FormulaCategory) -> str:
    """Get color for a formula category."""
    colors = {
        FormulaCategory.INPUT: "#4CAF50",       # Green
        FormulaCategory.DEVELOPMENT: "#2196F3",  # Blue
        FormulaCategory.FINANCING: "#9C27B0",    # Purple
        FormulaCategory.REVENUE: "#FF9800",      # Orange
        FormulaCategory.OPERATIONS: "#00BCD4",   # Cyan
        FormulaCategory.INVESTMENT: "#F44336",   # Red
        FormulaCategory.RETURNS: "#E91E63",      # Pink
    }
    return colors.get(category, "#9E9E9E")


def _build_dependency_graph() -> Tuple[Dict[str, FormulaDefinition], List[Tuple[str, str]]]:
    """Build the dependency graph from the formula registry.

    Returns:
        Tuple of (nodes dict, edges list)
        - nodes: field_path -> FormulaDefinition
        - edges: list of (from_field, to_field) tuples
    """
    all_formulas = FormulaRegistry.get_all()
    nodes = {}
    edges = []

    for field_path, formula in all_formulas.items():
        nodes[field_path] = formula
        # Add edges from each input to this formula
        for input_path in formula.inputs:
            edges.append((input_path, field_path))

    return nodes, edges


def _hierarchical_layout(
    nodes: Dict[str, FormulaDefinition],
    edges: List[Tuple[str, str]]
) -> Dict[str, Tuple[float, float]]:
    """Compute hierarchical layout positions for nodes.

    Arranges nodes by category in layers from left to right.
    """
    # Define layer order (left to right)
    layer_order = [
        FormulaCategory.INPUT,
        FormulaCategory.DEVELOPMENT,
        FormulaCategory.FINANCING,
        FormulaCategory.REVENUE,
        FormulaCategory.OPERATIONS,
        FormulaCategory.INVESTMENT,
        FormulaCategory.RETURNS,
    ]

    # Group nodes by category
    by_category: Dict[FormulaCategory, List[str]] = {cat: [] for cat in layer_order}
    for field_path, formula in nodes.items():
        if formula.category in by_category:
            by_category[formula.category].append(field_path)

    # Compute positions
    positions = {}
    x_spacing = 1.5

    for layer_idx, category in enumerate(layer_order):
        category_nodes = by_category[category]
        n_nodes = len(category_nodes)

        if n_nodes == 0:
            continue

        x = layer_idx * x_spacing

        # Distribute nodes vertically
        y_spacing = 1.0
        y_start = -(n_nodes - 1) * y_spacing / 2

        for node_idx, field_path in enumerate(sorted(category_nodes)):
            y = y_start + node_idx * y_spacing
            positions[field_path] = (x, y)

    return positions


def _create_plotly_graph(
    nodes: Dict[str, FormulaDefinition],
    edges: List[Tuple[str, str]],
    positions: Dict[str, Tuple[float, float]],
    highlight_path: Optional[str] = None,
) -> go.Figure:
    """Create a Plotly figure for the dependency graph."""

    # Filter to only nodes with positions (those in our categories)
    valid_nodes = {k: v for k, v in nodes.items() if k in positions}
    valid_edges = [(f, t) for f, t in edges if f in positions and t in positions]

    # Create edge traces
    edge_x = []
    edge_y = []

    for from_node, to_node in valid_edges:
        x0, y0 = positions[from_node]
        x1, y1 = positions[to_node]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces (one per category for legend)
    node_traces = []

    for category in FormulaCategory:
        category_nodes = [fp for fp, f in valid_nodes.items() if f.category == category]
        if not category_nodes:
            continue

        node_x = [positions[fp][0] for fp in category_nodes]
        node_y = [positions[fp][1] for fp in category_nodes]
        node_text = [valid_nodes[fp].name for fp in category_nodes]
        hover_text = [
            f"<b>{valid_nodes[fp].name}</b><br>"
            f"Path: {fp}<br>"
            f"Formula: {valid_nodes[fp].formula}<br>"
            f"Inputs: {len(valid_nodes[fp].inputs)}"
            for fp in category_nodes
        ]

        # Highlight selected node
        sizes = []
        for fp in category_nodes:
            if highlight_path and fp == highlight_path:
                sizes.append(20)
            else:
                sizes.append(12)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            name=category.value,
            text=node_text,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                size=sizes,
                color=_get_category_color(category),
                line=dict(width=1, color='white')
            )
        )
        node_traces.append(node_trace)

    # Create figure
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text='Formula Dependency Graph',
                x=0.5,
                xanchor='center'
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
            height=600,
        )
    )

    return fig


def render_dependency_graph_view() -> None:
    """Render the interactive dependency graph visualization."""

    st.header("Formula Dependency Graph")
    st.markdown(
        "Explore how calculations flow through the model. "
        "Each node is a formula, and edges show dependencies."
    )

    # Build the graph
    nodes, edges = _build_dependency_graph()
    positions = _hierarchical_layout(nodes, edges)

    # Category filter
    st.markdown("#### Filter by Category")

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_categories = st.multiselect(
            "Show categories",
            options=[cat.value for cat in FormulaCategory],
            default=[cat.value for cat in FormulaCategory],
            key="dep_graph_categories"
        )

    with col2:
        # Node selector for highlighting
        all_paths = sorted(nodes.keys())
        highlight_path = st.selectbox(
            "Highlight formula",
            options=["(none)"] + all_paths,
            key="dep_graph_highlight"
        )
        if highlight_path == "(none)":
            highlight_path = None

    # Filter nodes by selected categories
    category_set = {FormulaCategory(c) for c in selected_categories}
    filtered_nodes = {
        fp: f for fp, f in nodes.items()
        if f.category in category_set
    }
    filtered_edges = [
        (f, t) for f, t in edges
        if f in filtered_nodes and t in filtered_nodes
    ]
    filtered_positions = {
        fp: pos for fp, pos in positions.items()
        if fp in filtered_nodes
    }

    # Create and display the graph
    fig = _create_plotly_graph(
        filtered_nodes,
        filtered_edges,
        filtered_positions,
        highlight_path
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show details for highlighted node
    if highlight_path and highlight_path in nodes:
        formula = nodes[highlight_path]

        st.markdown("---")
        st.markdown(f"### {formula.name}")
        st.code(formula.formula, language=None)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Inputs (dependencies):**")
            if formula.inputs:
                for inp in formula.inputs:
                    if inp in nodes:
                        st.write(f"- {nodes[inp].name} (`{inp}`)")
                    else:
                        st.write(f"- `{inp}` (raw input)")
            else:
                st.write("No dependencies (base input)")

        with col2:
            st.markdown("**Used by (dependents):**")
            dependents = FormulaRegistry.get_dependents(highlight_path)
            if dependents:
                for dep in dependents:
                    if dep in nodes:
                        st.write(f"- {nodes[dep].name} (`{dep}`)")
            else:
                st.write("Not used by other formulas (terminal)")

    # Stats
    st.markdown("---")
    st.markdown("#### Graph Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Formulas", len(filtered_nodes))
    with col2:
        st.metric("Dependencies", len(filtered_edges))
    with col3:
        # Count terminal nodes (no outgoing edges)
        sources = {e[0] for e in filtered_edges}
        targets = {e[1] for e in filtered_edges}
        terminals = len(targets - sources)
        st.metric("Terminal Outputs", terminals)
