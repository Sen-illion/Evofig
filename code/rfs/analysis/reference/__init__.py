"""Reference-image analysis helpers."""

from .style_analyzer import analyze_reference_image
from .figure_source import reference_image_path
from .dataset_selector import select_positive_framework_reference
from .exemplars import resolve_project_exemplars, select_gap_reference, select_reference_exemplars
from .library_learning import analyze_reference_figures
from .style_learning import learn_reference_style
from .ppt_layout import analyze_reference_ppt_layout

__all__ = ["analyze_reference_figures", "analyze_reference_image", "analyze_reference_ppt_layout", "learn_reference_style", "reference_image_path", "resolve_project_exemplars", "select_gap_reference", "select_positive_framework_reference", "select_reference_exemplars"]
