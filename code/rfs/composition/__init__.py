"""Deterministic editable-figure composition and preview rendering."""

from .pptx import compile_ppt
from .preview import render_rebuild_preview
from .pptx_render import render_deck_preview, render_pptx_page
from .arrow_routing import style_and_route_arrows
from .text_layout import fit_text_program_globally
from .node_layout import arrange_node_contents
from .overlay_geometry import guard_overlay_geometry, keep_panels_below_title_band
from .layout_solver import solve_layout_constraints
from .joint_geometry import resolve_joint_geometry
from .direct_connectors import apply_connector_route_selections, apply_memory_routing_constraints, resolve_direct_connectors
from .layout_repairs import CURATED_LAYOUT_RULES, apply_direct_layout_repairs, apply_layout_constraints
from .typography import optimize_direct_typography
from .semantic_figure import build_semantic_figure_program
from .editable_graph import build_editable_graph
from .color_application import apply_color_theme_to_program
from .figure_edit import run_figure_edit

__all__ = ["apply_color_theme_to_program", "arrange_node_contents", "guard_overlay_geometry", "keep_panels_below_title_band", "CURATED_LAYOUT_RULES", "apply_direct_layout_repairs", "apply_layout_constraints", "run_figure_edit", "apply_connector_route_selections", "apply_memory_routing_constraints", "build_editable_graph", "build_semantic_figure_program", "compile_ppt", "fit_text_program_globally", "optimize_direct_typography", "render_deck_preview", "render_pptx_page", "render_rebuild_preview", "resolve_direct_connectors", "resolve_joint_geometry", "solve_layout_constraints", "style_and_route_arrows"]
