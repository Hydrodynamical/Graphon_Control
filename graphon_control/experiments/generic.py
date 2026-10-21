"""General time/label control experiment using the shared numerical core.

The paper coefficient is beta, not the prototype's former beta/2. Optional
uniform, terminal-variance and scheduled-target terms are explicit extensions.
"""
import argparse
import dataclasses
import inspect
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from ..controls import FourierFeatures, MLP, TimeLabelGraphonControl
from ..dynamics import rollout_dynamics_w1
from ..objectives import (empirical_mean, empirical_variance, graphon_l2_penalty,
                         graphon_uniform_penalty, uniform_graphon_target,
                         resolve_target_schedule, sampled_cost_JNdt, trajectory_cost)
from ..initial_conditions import (sample_x0, sample_x0_structured, sample_x0_two_clusters,
                                  sample_x0_linear_label, sample_x0_linear_sine)
from ..interactions import phi_default, phi_linear
from ..training import train_controller
from ..validation import validate_common_config
from ..artifacts import create_run_directory, write_run_provenance, save_checkpoint
from ..cli import add_common_arguments, configuration_from_args


@dataclass
class TrainConfig:
    # discretization/noise parameters
    N: int = 64                     # number of agents / label grid points
    T: float = 1.0                  # total time horizon
    dt: float = 0.02                # time step for Euler-Maruyama simulation and cost discretization
    sigma: float = 0.15             # noise scale in dynamics

    # cost parameters
    beta: float = 1e-2              # control regularization coefficient in J^{N,Δt}
    eta_uniform: float = 0.0        # coefficient for uniform-graphon regularizer
    regularizer_mode: str = "l2"    # one of: "l2", "uniform", "l2_plus_uniform"
    lambda_terminal: float = 1.0    # weight on terminal variance penalty
    gamma_target: float = 0.0       # weight on terminal mean-tracking penalty
    x_target: float = 0.0           # target for the terminal empirical mean
    target_times: List[float] = dataclasses.field(default_factory=list)    # additional checkpoint times in [0, T]
    target_values: List[float] = dataclasses.field(default_factory=list)   # target mean value at each checkpoint time
    target_weights: List[float] = dataclasses.field(default_factory=list)  # optional weights for checkpoint targets; defaults to gamma_target if omitted

    # graphon architecture
    enforce_symmetry: bool = False   # if True, enforce W_ij = W_ji exactly
    positive_map: str = "softplus"   # nonnegativity map for symmetric mode: "softplus" or "exp"

    # Monte Carlo / optimization
    batch_size: int = 32
    num_steps: int = 1000
    lr: float = 1e-3
    grad_clip: Optional[float] = 1.0

    # network architecture
    num_fourier_freqs: int = 8    # m in the draft
    embed_dim: int = 64           # d in the draft
    hidden_embed: int = 128
    hidden_score: int = 128
    time_scale: Optional[float] = None       # rescales t before feeding to score net

    # loss weighting
    weighted_running_cost: bool = False  # if True, linearly ramp running state cost weight from ~0 to 1 over [0, T]

    # interaction scaling
    interaction_alpha: float = 1.0      # linear interaction gain; positive is attractive (consensus-seeking) under the paper convention φ(x_j - x_i)

    # reproducibility / device
    eval_batch_size: int = 128
    evaluation_seed: int = 1729
    score_bound: Optional[float] = None
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        validate_common_config(self)
        if self.regularizer_mode not in ("l2", "uniform", "l2_plus_uniform"):
            raise ValueError("unknown regularizer_mode")
        resolve_target_schedule(self.T, self.dt, self.gamma_target,
                                self.target_times, self.target_values, self.target_weights)


def train_algorithm1_w1(config: TrainConfig, phi=None,
                        x0_sampler=sample_x0_two_clusters):
    """Train an open-loop controller; samplers return (batch, agents)."""
    config.__post_init__()
    phi = partial(phi_default, alpha=config.interaction_alpha) if phi is None else phi
    torch.manual_seed(config.seed)
    xi = (torch.arange(config.N, device=config.device, dtype=torch.float32) + 0.5) / config.N
    model = TimeLabelGraphonControl(
        xi, config.num_fourier_freqs, config.embed_dim, config.hidden_embed,
        config.hidden_score, time_scale=config.T if config.time_scale is None else config.time_scale,
        enforce_symmetry=config.enforce_symmetry, positive_map=config.positive_map,
        score_bound=config.score_bound)

    def objective():
        loss, aux = sampled_cost_JNdt(
            model, x0_sampler(config.batch_size, xi), T=config.T, dt=config.dt,
            sigma=config.sigma, beta=config.beta, lambda_terminal=config.lambda_terminal,
            phi=phi, gamma_target=config.gamma_target, x_target=config.x_target,
            target_times=config.target_times, target_values=config.target_values,
            target_weights=config.target_weights, eta_uniform=config.eta_uniform,
            regularizer_mode=config.regularizer_mode,
            weighted_running_cost=config.weighted_running_cost)
        return {'total': loss, **aux}

    history, last = train_controller(model, objective, config.num_steps, config.lr,
                                    config.grad_clip, description='training')
    return model, history, xi, {key: val for key, val in last.items() if key != 'total'}


@torch.no_grad()
def evaluate_policy(model, xi, T, dt, sigma, batch_size,
                    phi=phi_default, x0_sampler=sample_x0_two_clusters):
    model.eval()
    return rollout_dynamics_w1(model, x0_sampler(batch_size, xi), T, dt, sigma, phi)


def save_run(out_root, config, history, final_aux, phi, x0_sampler,
             x0_sample, run_tag='', reserved_run_dir=None):
    """Persist configuration/provenance and training diagnostics to a new run."""
    run_dir = (create_run_directory(out_root, 'run', run_tag or None)
               if reserved_run_dir is None else Path(reserved_run_dir))
    if reserved_run_dir is None:
        write_run_provenance(run_dir, config, config.seed, config.evaluation_seed,
                             device=config.device)
    for filename, value in [('history.json', history), ('final_aux.json', final_aux)]:
        (run_dir / filename).write_text(json.dumps(value, indent=2) + '\n')
    try:
        source = inspect.getsource(phi)
    except (OSError, TypeError):
        source = repr(phi)
    summary = (f'Interaction: {getattr(phi, "__name__", repr(phi))}\n'
               f'Sampler: {getattr(x0_sampler, "__name__", repr(x0_sampler))}\n'
               f'Initial sample mean: {float(x0_sample.mean()):.6g}\n'
               f'Interaction source:\n{source}\n')
    (run_dir / 'run_summary.txt').write_text(summary)
    return str(run_dir)


@torch.no_grad()
def _archive_evaluation(run_dir, model, config, xi, x0_sample, phi, sampler, batch_size):
    xs, Ws = evaluate_policy(model, xi, config.T, config.dt, config.sigma,
                             batch_size, phi=phi, x0_sampler=sampler)
    schedule = resolve_target_schedule(config.T, config.dt, config.gamma_target,
                                       config.target_times, config.target_values, config.target_weights)
    parts = trajectory_cost(torch.stack(xs), torch.stack(Ws), dt=config.dt, beta=config.beta,
                            lambda_terminal=config.lambda_terminal, gamma_target=config.gamma_target,
                            x_target=config.x_target, target_schedule=schedule,
                            eta_uniform=config.eta_uniform, regularizer_mode=config.regularizer_mode,
                            weighted_running_cost=config.weighted_running_cost)
    metrics = {key: float(value) for key, value in parts.items()}
    Path(run_dir, 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    phi_grid = torch.linspace(-20, 20, 401, device=xi.device, dtype=xi.dtype)
    np.savez_compressed(Path(run_dir) / 'trajectories.npz',
                        xs=torch.stack(xs).cpu().numpy(), Ws=torch.stack(Ws).cpu().numpy(),
                        xi=xi.cpu().numpy(), x0_sample=x0_sample.cpu().numpy(),
                        phi_grid=phi_grid.cpu().numpy(), phi_values=phi(phi_grid).cpu().numpy())
    return xs, Ws


def run_experiment(config: TrainConfig, phi=None, x0_sampler=sample_x0_two_clusters,
                   out_root='out_1', run_tag='', eval_batch_size=None, make_plots=True):
    """Train, archive, evaluate and optionally render using the same shared core."""
    eval_batch_size = config.eval_batch_size if eval_batch_size is None else eval_batch_size
    if not isinstance(eval_batch_size, int) or isinstance(eval_batch_size, bool) or eval_batch_size < 1:
        raise ValueError('eval_batch_size must be a positive integer')
    config = dataclasses.replace(config, eval_batch_size=eval_batch_size)
    reserved = create_run_directory(out_root, 'run', run_tag or None)
    write_run_provenance(reserved, config, config.seed, config.evaluation_seed, device=config.device)
    phi = partial(phi_default, alpha=config.interaction_alpha) if phi is None else phi
    model, history, xi, final_aux = train_algorithm1_w1(config, phi, x0_sampler)
    torch.manual_seed(config.evaluation_seed)
    x0_sample = x0_sampler(config.batch_size, xi)
    run_dir = save_run(out_root, config, history, final_aux, phi, x0_sampler, x0_sample,
                       run_tag, reserved_run_dir=reserved)
    save_checkpoint(Path(run_dir) / 'model.pt', model)
    xs, Ws = _archive_evaluation(run_dir, model, config, xi, x0_sample, phi,
                                  x0_sampler, eval_batch_size)
    if make_plots:
        plot_run(run_dir)
    Path(run_dir, 'complete.json').write_text(json.dumps({'status': 'complete', 'plots_generated': make_plots}) + '\n')
    print(f'Results saved to: {run_dir}')
    return {'model': model, 'history': history, 'xi': xi, 'xs': xs, 'Ws': Ws, 'run_dir': run_dir}


def plot_run(run_dir):
    """Render saved numerical arrays without retraining or evaluating a callable."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ..plotting import (imshow_graphon, plot_graphon_diagnostics, heterogeneity_stats,
                            plot_state_histogram_over_time_imshow, plot_x0_samples,
                            plot_x0_density, save_graphon_gif)
    run_dir = Path(run_dir)
    cfg = TrainConfig(**json.loads((run_dir / 'config.json').read_text()))
    with np.load(run_dir / 'trajectories.npz', allow_pickle=False) as archive:
        data = {name: torch.as_tensor(value) for name, value in archive.items()}
    xs, Ws, xi, x0 = data['xs'], data['Ws'], data['xi'], data['x0_sample']
    lines = []
    for label, index in [('early', len(Ws) // 4), ('mid', len(Ws) // 2), ('late', 3 * len(Ws) // 4)]:
        W = Ws[index]
        lines.append(f'{label}: row sums {W.sum(-1).tolist()}; statistics {heterogeneity_stats(W)}')
        plot_graphon_diagnostics(cfg.N * W, title_prefix=f'{label}-time ', show=False)
        plt.gcf().savefig(run_dir / f'graphon_diag_{label}.png', dpi=150)
        plt.close(plt.gcf())
    (run_dir / 'eval_diagnostics.txt').write_text('\n'.join(lines) + '\n')
    fig, ax = plt.subplots(figsize=(7, 6))
    imshow_graphon(cfg.N * Ws[len(Ws) // 2], ax=ax, show=False)
    fig.savefig(run_dir / 'graphon_mid.png', dpi=150)
    plt.close(fig)
    plot_state_histogram_over_time_imshow(list(xs), dt=cfg.dt, show=False, x_target=cfg.x_target)
    plt.gcf().savefig(run_dir / 'state_histogram.png', dpi=150)
    plt.close(plt.gcf())
    for filename, plotter in [('ic_samples.png', plot_x0_samples), ('ic_density.png', plot_x0_density)]:
        fig, _ = plotter(xi, x0)
        fig.savefig(run_dir / filename, dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots()
    ax.plot(data['phi_grid'], data['phi_values'])
    ax.set(xlabel='state difference', ylabel='interaction')
    fig.savefig(run_dir / 'phi_plot.png', dpi=150)
    plt.close(fig)
    indices = np.linspace(0, len(Ws) - 1, min(60, len(Ws))).round().astype(int)
    save_graphon_gif([cfg.N * Ws[i] for i in indices], str(run_dir / 'graphon_time.gif'),
                     t_vals=[float(i * cfg.dt) for i in indices])

    complete = run_dir / "complete.json"
    if complete.exists():
        payload = json.loads(complete.read_text())
        payload["plots_generated"] = True
        complete.write_text(json.dumps(payload) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description='General time-and-label graphon control')
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        if args.replot:
            plot_run(args.replot)
            return
        cfg = configuration_from_args(TrainConfig, args)
        run_experiment(cfg, out_root=args.out_root, run_tag=args.tag, make_plots=not args.no_plots)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
