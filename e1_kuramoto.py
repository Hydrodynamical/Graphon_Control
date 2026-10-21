"""Compatibility entry point; implementation lives in graphon_control.experiments.kuramoto."""
from graphon_control.experiments.kuramoto import *  # noqa: F401,F403
from graphon_control.experiments.kuramoto import main

_LEGACY_PLOT_NAMES = ('fig_r_panels', 'fig_r_combined', 'fig_winding', 'fig_variance', 'fig_vxi', 'fig_kernel_strip', 'fig_state_evolution', 'fig_training_history', 'save_untwist_gif', 'ARM_LABELS', 'ARM_COLORS', 'PAPER_ARMS')

def __getattr__(name):
    if name in _LEGACY_PLOT_NAMES:
        from importlib import import_module
        return getattr(import_module("graphon_control.experiments.kuramoto_plots"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if __name__ == "__main__":
    main()
