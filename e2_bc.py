"""Compatibility entry point; implementation lives in graphon_control.experiments.bounded_confidence."""
from graphon_control.experiments.bounded_confidence import *  # noqa: F401,F403
from graphon_control.experiments.bounded_confidence import main

_LEGACY_PLOT_NAMES = ('fig_state_evolution', 'fig_opinion_traj', 'fig_kernel_strip', 'fig_cluster_count', 'fig_variance', 'fig_authority', 'fig_vxi', 'fig_phi', 'fig_training_history', 'save_opinion_gif', 'ARM_LABELS', 'ARM_COLORS')

def __getattr__(name):
    if name in _LEGACY_PLOT_NAMES:
        from importlib import import_module
        return getattr(import_module("graphon_control.experiments.bounded_confidence_plots"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if __name__ == "__main__":
    main()
