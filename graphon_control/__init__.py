"""Differentiable time-and-label graphon control, following the paper's conventions.

W is row-stochastic; the graphon density on a uniform N-grid is N*W.
Importing the numerical package does not load plotting or run experiments.
"""
from .controls import FourierFeatures, MLP, TimeLabelGraphonControl
from .dynamics import ConstantKernel, mean_field_kernel, precompute_Ws, rollout_with_Ws
from .objectives import paper_cost, trajectory_cost
from .diagnostics import vxi_discrete
from .training import train_controller

__all__ = ['FourierFeatures', 'MLP', 'TimeLabelGraphonControl', 'ConstantKernel',
           'mean_field_kernel', 'precompute_Ws', 'rollout_with_Ws', 'paper_cost',
           'trajectory_cost', 'vxi_discrete', 'train_controller']
__version__ = '0.1.0'
