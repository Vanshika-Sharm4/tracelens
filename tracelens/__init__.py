"""TraceLens: an ML inference profiler and bottleneck visualizer for PyTorch."""

from tracelens.analysis import Analysis, AnalysisConfig, Finding, analyze
from tracelens.cost import HARDWARE, Hardware
from tracelens.profiler import profile_callable
from tracelens.trace import Trace, parse_chrome_trace

__version__ = "0.1.0"
__all__ = [
    "Analysis",
    "AnalysisConfig",
    "Finding",
    "HARDWARE",
    "Hardware",
    "Trace",
    "analyze",
    "parse_chrome_trace",
    "profile_callable",
    "__version__",
]
