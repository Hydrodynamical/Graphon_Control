# model_1.py

import dataclasses
import datetime
import inspect
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

# PyTorch for modeling and training
import torch
import torch.nn as nn
import torch.nn.functional as F

# import vizualization utilities from model_1_viz.py
from tqdm import tqdm
from model_1_viz import imshow_graphon, plot_graphon_diagnostics, heterogeneity_stats, plot_state_histogram_over_time_imshow, plot_x0_samples, plot_x0_density, plot_phi, save_graphon_gif


# ============================================================
# Random run-tag generator
# ============================================================

_TAG_ADJECTIVES = [
    "amber", "arctic", "azure", "bold", "bright", "calm", "cardinal",
    "cobalt", "coral", "crimson", "crystal", "dark", "dawn", "dense", "dim",
    "dusty", "dusk", "ember", "faint", "firm", "flat", "fluid", "foggy",
    "frozen", "gilded", "golden", "green", "grey", "hollow", "hushed", "icy",
    "iron", "jade", "keen", "light", "lunar", "muted", "narrow", "noble",
    "ochre", "pale", "quiet", "rapid", "rigid", "rough", "ruby", "russet",
    "sandy", "scarlet", "sharp", "silent", "silver", "sleek", "slow", "solar",
    "sparse", "stark", "stern", "still", "stone", "stormy", "subtle", "swift",
    "tawny", "teal", "thin", "tidal", "tiny", "topaz", "turbid", "twilight",
    "vast", "velvet", "vivid", "warm", "white", "wild", "windy", "wiry",
]

_TAG_NOUNS = [
    "basin", "beacon", "bloom", "boulder", "brook", "canopy", "canyon",
    "cliff", "cloud", "coast", "comet", "coral", "crater", "creek", "crest",
    "delta", "dune", "eddy", "fern", "field", "fjord", "flare", "flint",
    "fog", "forest", "frost", "gale", "gap", "geyser", "glacier", "glen",
    "gorge", "grove", "gulf", "heath", "inlet", "island", "kelp", "knoll",
    "lagoon", "larch", "ledge", "marsh", "meadow", "mesa", "moor", "moss",
    "nebula", "peak", "pine", "plain", "pond", "pool", "prism", "ravine",
    "reef", "ridge", "rift", "river", "rock", "shore", "slope", "spring",
    "storm", "stream", "summit", "swamp", "tide", "timber", "torrent", "trail",
    "vale", "valley", "veil", "void", "wave", "wisp",
]


def _random_run_tag() -> str:
    import random
    return f"{random.choice(_TAG_ADJECTIVES)}_{random.choice(_TAG_NOUNS)}"


# ============================================================
# Configuration
# ============================================================

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
    time_scale: float = 1.0       # rescales t before feeding to score net

    # loss weighting
    weighted_running_cost: bool = False  # if True, linearly ramp running state cost weight from ~0 to 1 over [0, T]

    # interaction scaling
    interaction_alpha: float = -1.0     # linear interaction gain; negative compensates the current sign convention and guides toward consensus

    # reproducibility / device
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utility: interaction nonlinearity φ
# ============================================================

def phi_default(y: torch.Tensor, alpha: float = -1.0) -> torch.Tensor:
    """
    Default linear interaction φ(y) = -y.

    With the current rollout sign convention, the negative sign yields
    consensus-seeking drift.
    """
    return alpha * torch.tanh(y)


def phi_linear(y: torch.Tensor, alpha: float = -1.0) -> torch.Tensor:
    """
    Linear interaction φ(y) = alpha * y.

    With the current rollout sign convention, alpha < 0 yields
    consensus-seeking drift.
    """
    return alpha * y


# ============================================================
# Fourier features for labels ξ in [0,1]
# ψ(ξ) = (sin(2π 2^k ξ), cos(2π 2^k ξ))_{k=0}^{m-1}
# ============================================================

class FourierFeatures(nn.Module):
    def __init__(self, num_freqs: int):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs: torch.Tensor
        freqs = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (...,) in [0,1]
        returns: (..., 2*num_freqs)
        """
        x = x.unsqueeze(-1)  # (..., 1)
        angles = 2.0 * math.pi * x * self.freqs  # (..., m)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# ============================================================
# Simple MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Graphon control architecture (Model 1: time + label)
#
# enforce_symmetry=False (directed):
#   receiver h_i = φθ(ψ(ξ_i)),  sender h̃_j = φ̃θ(ψ(ξ_j))
#   score s_ij(t) = gθ(t, h_i, h̃_j),  W = masked row-softmax
#
# enforce_symmetry=True (undirected):
#   shared h_i = φθ(ψ(ξ_i))
#   A_ij = gθ(t, h_i, h_j),  S = 0.5*(A + Aᵀ),  W = positive_map(S) / avg_row_sum
# ============================================================

class TimeLabelGraphonControl(nn.Module):
    def __init__(
        self,
        xi: torch.Tensor,
        num_fourier_freqs: int,
        embed_dim: int,
        hidden_embed: int,
        hidden_score: int,
        time_scale: float = 1.0,
        enforce_symmetry: bool = False,
        positive_map: str = "softplus",
    ):
        super().__init__()
        self.xi: torch.Tensor
        self.psi_xi_cached: torch.Tensor
        self.diag_mask: torch.Tensor
        self.register_buffer("xi", xi)          # (N,)
        self.N = xi.numel()
        self.time_scale = time_scale
        self.enforce_symmetry = enforce_symmetry
        self.positive_map = positive_map

        self.ff = FourierFeatures(num_fourier_freqs)
        ff_dim = self.ff.out_dim

        # Symmetric mode:  single shared embedding  (h_i = h_j basis, symmetry by construction)
        # Asymmetric mode: separate receiver/sender embeddings (directed attention, row-softmax)
        self.embed_net:    Optional[MLP] = MLP(ff_dim, hidden_embed, embed_dim, depth=2) if enforce_symmetry else None
        self.receiver_net: Optional[MLP] = None if enforce_symmetry else MLP(ff_dim, hidden_embed, embed_dim, depth=2)
        self.sender_net:   Optional[MLP] = None if enforce_symmetry else MLP(ff_dim, hidden_embed, embed_dim, depth=2)

        # Score net gθ : R^{1 + 2d} -> R  (shared across both modes)
        self.score_net = MLP(1 + 2 * embed_dim, hidden_score, 1, depth=2)

        # Precompute label features / embeddings at the fixed label grid
        psi_xi = self.ff(self.xi)                          # (N, 2m)
        self.register_buffer("psi_xi_cached", psi_xi)

        # mask for self-interaction
        mask = torch.eye(self.N, dtype=torch.bool, device=xi.device)
        self.register_buffer("diag_mask", mask)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: scalar tensor or shape (B,)
        returns: (N, N) if scalar t, (B, N, N) if batched t

        enforce_symmetry=False  →  directed, row-stochastic  (each row sums to 1)
        enforce_symmetry=True   →  undirected, symmetric,    average row sum = 1
        """
        N = self.N

        if t.ndim == 0:
            t = t.unsqueeze(0)

        B = t.shape[0]
        t_in = (t / self.time_scale).reshape(B, 1, 1, 1).expand(B, N, N, 1)

        if self.enforce_symmetry:
            assert self.embed_net is not None
            h = self.embed_net(self.psi_xi_cached)              # (N, d)
            d = h.shape[-1]
            h_i = h.view(1, N, 1, d).expand(B, N, N, d)
            h_j = h.view(1, 1, N, d).expand(B, N, N, d)

            A = self.score_net(torch.cat([t_in, h_i, h_j], dim=-1)).squeeze(-1)  # (B, N, N)
            A = 0.5 * (A + A.transpose(-1, -2))                 # exact symmetry

            W = torch.exp(A.clamp(max=20.0)) if self.positive_map == "exp" else F.softplus(A)
            W = W.masked_fill(self.diag_mask.unsqueeze(0), 0.0)

            # Normalize by avg row sum — scalar per batch element, preserves symmetry
            avg_row_sum = W.sum(dim=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-8)
            W = W / avg_row_sum

        else:
            assert self.receiver_net is not None and self.sender_net is not None
            receiver_h = self.receiver_net(self.psi_xi_cached)  # (N, d)
            sender_h   = self.sender_net(self.psi_xi_cached)    # (N, d)
            d = receiver_h.shape[-1]
            h_i = receiver_h.view(1, N, 1, d).expand(B, N, N, d)
            h_j = sender_h.view(1, 1, N, d).expand(B, N, N, d)

            scores = self.score_net(torch.cat([t_in, h_i, h_j], dim=-1)).squeeze(-1)  # (B, N, N)
            scores = scores.masked_fill(self.diag_mask.unsqueeze(0), -1e9)
            W = F.softmax(scores, dim=-1)
            W = W.masked_fill(self.diag_mask.unsqueeze(0), 0.0)

        if W.shape[0] == 1:
            return W[0]
        return W


# ============================================================
# Simulation of dynamics (5)
# x_i^{n+1} = x_i^n + Δt (1/N) Σ_j W_ij(t_n) φ(x_j^n - x_i^n) + σ ΔB_i^n
# ============================================================

def rollout_dynamics_w1(
    model: TimeLabelGraphonControl,
    x0: torch.Tensor,
    T: float,
    dt: float,
    sigma: float,
    phi: Callable[[torch.Tensor], torch.Tensor],
):
    """
    x0: (B, N)
    returns:
        xs: list of length NT+1, each entry (B, N)
        Ws: list of length NT, each entry (N, N)   # W1 depends only on time
    """
    device = x0.device
    B, N = x0.shape
    NT = int(round(T / dt))

    x = x0
    xs = [x]
    Ws = []

    sqrt_dt = math.sqrt(dt)

    for n in range(NT):
        t_n = torch.tensor(n * dt, device=device, dtype=x0.dtype)
        W = model(t_n)  # (N, N)
        Ws.append(W)

        # pairwise differences x_j - x_i, shape (B, N, N)
        diff = x.unsqueeze(1) * 0.0  # dummy for shape hint
        diff = x.unsqueeze(1)  # (B, 1, N)
        diff = x.unsqueeze(1).expand(B, N, N) - x.unsqueeze(2).expand(B, N, N)
        # above is x_i - x_j, so flip sign:
        diff = -diff  # now diff[b, i, j] = x_j - x_i

        interaction = phi(diff)                      # (B, N, N)
        drift = (W.unsqueeze(0) * interaction).sum(dim=-1)          # (B, N)

        noise = sigma * sqrt_dt * torch.randn(B, N, device=device, dtype=x0.dtype)
        x = x + dt * drift + noise
        xs.append(x)

    return xs, Ws


# ============================================================
# Cost J^{N,Δt}(θ)
# Σ_n Δt [ Var(x^n) + (β/2)||W(t_n)||^2 ]
# with Var(x) = (1/N)Σ_k x_k^2 - ( (1/N)Σ_k x_k )^2
# ============================================================

def empirical_variance(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, N)
    returns: (B,)
    """
    mean = x.mean(dim=-1)
    second_moment = (x ** 2).mean(dim=-1)
    return second_moment - mean ** 2


def empirical_mean(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, N)
    returns: (B,)
    """
    return x.mean(dim=-1)


def resolve_target_schedule(
    T: float,
    dt: float,
    gamma_target: float,
    target_times: List[float],
    target_values: List[float],
    target_weights: List[float],
) -> List[Dict[str, float]]:
    """
    Resolve user-specified checkpoint targets onto the discrete simulation grid.

    Each target time is mapped to the nearest rollout index via round(t / dt).
    """
    if len(target_times) != len(target_values):
        raise ValueError(
            "target_times and target_values must have the same length; "
            f"got {len(target_times)} and {len(target_values)}"
        )

    if target_weights and len(target_weights) != len(target_times):
        raise ValueError(
            "target_weights must be empty or have the same length as target_times; "
            f"got {len(target_weights)} and {len(target_times)}"
        )

    if not target_times:
        return []

    resolved_weights = target_weights
    if not resolved_weights:
        if gamma_target == 0.0:
            raise ValueError(
                "target_weights is empty and gamma_target is 0.0, so scheduled targets would carry no weight. "
                "Set target_weights explicitly or use a nonzero gamma_target."
            )
        resolved_weights = [gamma_target] * len(target_times)

    NT = int(round(T / dt))
    schedule: List[Dict[str, float]] = []
    for target_time, target_value, target_weight in zip(target_times, target_values, resolved_weights):
        if target_time < 0.0 or target_time > T:
            raise ValueError(f"target time {target_time} must lie in [0, T={T}]")

        target_index = int(round(target_time / dt))
        target_index = max(0, min(target_index, NT))
        snapped_time = target_index * dt
        schedule.append(
            {
                "index": float(target_index),
                "time": snapped_time,
                "value": float(target_value),
                "weight": float(target_weight),
            }
        )

    return schedule


def graphon_l2_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Discrete approximation of ||w||^2_{L^2(I^2)} when W is a row-stochastic
    weight matrix representing quadrature weights rather than raw graphon values.
    Since w_ij ~ N * W_ij, we use sum(W^2), not mean(W^2).
    """
    return (W ** 2).sum()


def uniform_graphon_target(
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return the discrete uniform graphon compatible with the zero-diagonal,
    row-stochastic architecture used here.

    The learned W has W_ii = 0 and each row sums to 1, so the N off-diagonal
    entries in each row share a total weight of 1.  The correct uniform target
    is therefore U_ij = 1/(N-1) for i != j, U_ii = 0
    — NOT the all-ones matrix, which would violate row normalization.
    """
    if N <= 1:
        return torch.zeros(N, N, device=device, dtype=dtype)
    U = torch.full((N, N), 1.0 / (N - 1), device=device, dtype=dtype)
    U.fill_diagonal_(0.0)
    return U


def graphon_uniform_penalty(W: torch.Tensor) -> torch.Tensor:
    """
    Discrete approximation of ||w - u||^2_{L^2(I^2)} in the row-stochastic
    representation. The correct scale is sum((W - U)^2), not mean((W - U)^2).
    """
    N = W.shape[0]
    U = uniform_graphon_target(N, device=W.device, dtype=W.dtype)
    return ((W - U) ** 2).sum()


def sampled_cost_JNdt(
    model: TimeLabelGraphonControl,
    x0: torch.Tensor,
    T: float,
    dt: float,
    sigma: float,
    beta: float,
    lambda_terminal: float,
    phi: Callable[[torch.Tensor], torch.Tensor],
    gamma_target: float = 0.0,
    x_target: float = 0.0,
    target_times: Optional[List[float]] = None,
    target_values: Optional[List[float]] = None,
    target_weights: Optional[List[float]] = None,
    eta_uniform: float = 0.0,
    regularizer_mode: str = "l2",
    weighted_running_cost: bool = False,
):
    """
    Monte Carlo sampled objective corresponding to Algorithm 1.
    x0: (B, N)

    regularizer_mode controls which graphon penalties are active:
      "l2"             — only (beta/2)||W||^2 dt
      "uniform"        — only (eta_uniform/2)||W - U||^2 dt
      "l2_plus_uniform"— both penalties together

    returns:
        loss: scalar
        aux: dict with diagnostics
    """
    if regularizer_mode not in ("l2", "uniform", "l2_plus_uniform"):
        raise ValueError(
            f"regularizer_mode must be 'l2', 'uniform', or 'l2_plus_uniform'; "
            f"got '{regularizer_mode}'"
        )

    xs, Ws = rollout_dynamics_w1(model, x0, T=T, dt=dt, sigma=sigma, phi=phi)

    device = x0.device
    dtype = x0.dtype
    running_state_cost:   torch.Tensor = torch.zeros((), device=device, dtype=dtype)
    running_l2_reg_cost: torch.Tensor = torch.zeros((), device=device, dtype=dtype)
    running_uniform_cost: torch.Tensor = torch.zeros((), device=device, dtype=dtype)
    scheduled_target_cost: torch.Tensor = torch.zeros((), device=device, dtype=dtype)

    resolved_schedule = resolve_target_schedule(
        T=T,
        dt=dt,
        gamma_target=gamma_target,
        target_times=target_times or [],
        target_values=target_values or [],
        target_weights=target_weights or [],
    )

    NT = len(Ws)
    for n in range(NT):
        x_n = xs[n]           # (B, N)
        W_n = Ws[n]           # (N, N)

        var_n         = empirical_variance(x_n).mean()
        l2_reg_n      = graphon_l2_penalty(W_n)
        uniform_reg_n = graphon_uniform_penalty(W_n)

        w_n = (n + 1) * dt / T if weighted_running_cost else 1.0
        running_state_cost = running_state_cost + dt * w_n * var_n

        if regularizer_mode == "l2":
            running_l2_reg_cost = running_l2_reg_cost + dt * (beta / 2.0) * l2_reg_n
        elif regularizer_mode == "uniform":
            running_uniform_cost = running_uniform_cost + dt * (eta_uniform / 2.0) * uniform_reg_n
        else:  # "l2_plus_uniform"
            running_l2_reg_cost = running_l2_reg_cost + dt * (beta / 2.0) * l2_reg_n
            running_uniform_cost = running_uniform_cost + dt * (eta_uniform / 2.0) * uniform_reg_n

    x_T = xs[-1]                                   # (B, N)
    terminal_var = empirical_variance(x_T).mean()

    mean_T = empirical_mean(x_T)                   # (B,)
    target_tracking = ((mean_T - x_target) ** 2).mean()

    for target_spec in resolved_schedule:
        target_index = int(target_spec["index"])
        mean_at_target = empirical_mean(xs[target_index])
        scheduled_target_cost = scheduled_target_cost + target_spec["weight"] * ((mean_at_target - target_spec["value"]) ** 2).mean()

    terminal_var_cost = lambda_terminal * terminal_var
    terminal_target_cost = gamma_target * target_tracking
    terminal_cost = terminal_var_cost + terminal_target_cost

    total = running_state_cost + running_l2_reg_cost + running_uniform_cost + terminal_cost + scheduled_target_cost

    aux = {
        "total_cost": float(total.detach().cpu()),
        "state_cost": float(running_state_cost.detach().cpu()),
        "l2_reg_cost": float(running_l2_reg_cost.detach().cpu()),
        "uniform_cost": float(running_uniform_cost.detach().cpu()),
        "scheduled_target_cost": float(scheduled_target_cost.detach().cpu()),
        "terminal_var": float(terminal_var.detach().cpu()),
        "terminal_mean": float(mean_T.mean().detach().cpu()),
        "target_tracking": float(target_tracking.detach().cpu()),
        "terminal_var_cost": float(terminal_var_cost.detach().cpu()),
        "terminal_target_cost": float(terminal_target_cost.detach().cpu()),
        "terminal_cost": float(terminal_cost.detach().cpu()),
    }
    return total, aux


# ============================================================
# Sampling initial conditions
# ============================================================

def sample_x0(
    batch_size: int,
    N: int,
    mean: float,
    std: float,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return mean + std * torch.randn(batch_size, N, device=device, dtype=dtype)


def sample_x0_structured(
    batch_size: int,
    xi: torch.Tensor,
    noise_std: float = 0.2,
) -> torch.Tensor:
    """
    xi: (N,)
    Sinusoidal base profile with additive noise.
    """
    base = 1.5 * torch.sin(2 * torch.pi * xi) + 0.8 * torch.cos(4 * torch.pi * xi)
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device)


def sample_x0_two_clusters(
    batch_size: int,
    xi: torch.Tensor,
    noise_std: float = 0.15,
) -> torch.Tensor:
    """
    xi: (N,)
    Agents with xi < 0.5 start near -1, the rest near +1.
    """
    base = torch.where(xi < 0.5, torch.tensor(-1.0, device=xi.device), torch.tensor(1.0, device=xi.device))
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device)


def sample_x0_linear_label(
    batch_size: int,
    xi: torch.Tensor,
    noise_std: float = 0.15,
) -> torch.Tensor:
    """
    xi: (N,)
    Base profile 20*(ξ - 1/2) with additive Gaussian noise.
    Agents with higher labels start at higher states.
    """
    base = 20.0 * (xi - 0.5)
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device)


def sample_x0_linear_sine(
    batch_size: int,
    xi: torch.Tensor,
    amplitude: float = 7.0,
    n_periods: float = 5.0,
    noise_std: float = 1.0,
) -> torch.Tensor:
    """
    xi: (N,)
    Base profile: 20*(ξ - 1/2) + amplitude * sin(2π * n_periods * ξ)
    with additive Gaussian noise. Combines a linear label-dependent slope
    with multi-oscillation sinusoidal structure.
    """
    base = 20.0 * (xi - 0.5) + amplitude * torch.sin(2 * math.pi * n_periods * xi)
    return base.unsqueeze(0) + noise_std * torch.randn(batch_size, xi.numel(), device=xi.device, dtype=xi.dtype)


# ============================================================
# Training loop = Algorithm 1
# Pick sample S = (X0, ΔW), compute sampled cost, backprop, update θ
# ============================================================

def train_algorithm1_w1(
    config: TrainConfig,
    phi: Callable[[torch.Tensor], torch.Tensor] = phi_default,
    x0_sampler: Callable[[int, torch.Tensor], torch.Tensor] = sample_x0_two_clusters,
):
    torch.manual_seed(config.seed)

    device = torch.device(config.device)
    dtype = torch.float32

    N = config.N

    # label discretization ξ_i = (i + 1/2)/N
    xi = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N

    model = TimeLabelGraphonControl(
        xi=xi,
        num_fourier_freqs=config.num_fourier_freqs,
        embed_dim=config.embed_dim,
        hidden_embed=config.hidden_embed,
        hidden_score=config.hidden_score,
        time_scale=max(config.T, 1e-8),
        enforce_symmetry=config.enforce_symmetry,
        positive_map=config.positive_map,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: Dict[str, List[float]] = {
        "loss": [],
        "state_cost": [],
        "l2_reg_cost": [],
        "uniform_cost": [],
        "scheduled_target_cost": [],
        "terminal_var": [],
        "terminal_cost": [],
        "terminal_mean": [],
        "target_tracking": [],
        "terminal_var_cost": [],
        "terminal_target_cost": [],
    }

    pbar = tqdm(range(config.num_steps), desc="training", unit="step")
    for _ in pbar:
        model.train()
        optimizer.zero_grad()

        # New Monte Carlo sample S each iteration:
        # initial states and Brownian increments are generated inside sampled_cost_JNdt
        x0 = x0_sampler(config.batch_size, xi)

        loss, aux = sampled_cost_JNdt(
            model=model,
            x0=x0,
            T=config.T,
            dt=config.dt,
            sigma=config.sigma,
            beta=config.beta,
            lambda_terminal=config.lambda_terminal,
            phi=phi,
            gamma_target=config.gamma_target,
            x_target=config.x_target,
            target_times=config.target_times,
            target_values=config.target_values,
            target_weights=config.target_weights,
            eta_uniform=config.eta_uniform,
            regularizer_mode=config.regularizer_mode,
            weighted_running_cost=config.weighted_running_cost,
        )

        loss.backward()

        if config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        history["loss"].append(aux["total_cost"])
        history["state_cost"].append(aux["state_cost"])
        history["l2_reg_cost"].append(aux["l2_reg_cost"])
        history["uniform_cost"].append(aux["uniform_cost"])
        history["scheduled_target_cost"].append(aux["scheduled_target_cost"])
        history["terminal_var"].append(aux["terminal_var"])
        history["terminal_cost"].append(aux["terminal_cost"])
        history["terminal_mean"].append(aux["terminal_mean"])
        history["target_tracking"].append(aux["target_tracking"])
        history["terminal_var_cost"].append(aux["terminal_var_cost"])
        history["terminal_target_cost"].append(aux["terminal_target_cost"])
        last_aux = aux

        pbar.set_postfix(
            loss=f"{aux['total_cost']:.4f}",
            run=f"{aux['state_cost']:.4f}",
            term_var=f"{aux['terminal_var']:.4f}",
            term_mean=f"{aux['terminal_mean']:.4f}",
            target=f"{aux['terminal_target_cost'] + aux['scheduled_target_cost']:.4f}",
            reg=f"{aux['l2_reg_cost'] + aux['uniform_cost']:.4f}",
        )

    return model, history, xi, last_aux


# ============================================================
# Run persistence: save all artifacts to out_root/run_<timestamp>/
# ============================================================

def save_run(
    out_root: str,
    config: TrainConfig,
    history: Dict[str, List[float]],
    final_aux: dict,
    phi: Callable,
    x0_sampler: Callable,
    x0_sample: torch.Tensor,
    run_tag: str = "",
) -> str:
    """
    Save config, history, final diagnostics, and a human-readable summary
    to a timestamped directory under out_root.

    x0_sample: (B, N) — one representative initial batch for statistics.
    Returns the path of the created run directory.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"run_{ts}" + (f"_{run_tag}" if run_tag else "")
    run_dir = os.path.join(out_root, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # config.json
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    # history.json
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # final_aux.json
    with open(os.path.join(run_dir, "final_aux.json"), "w") as f:
        json.dump(final_aux, f, indent=2)

    # x0 statistics
    x0_np = x0_sample.detach().cpu()
    x0_stats = {
        "mean": float(x0_np.mean()),
        "std":  float(x0_np.std()),
        "min":  float(x0_np.min()),
        "max":  float(x0_np.max()),
    }

    # phi source (best-effort)
    try:
        phi_source = inspect.getsource(phi)
    except (OSError, TypeError):
        phi_source = "<source unavailable>"

    # run_summary.txt
    cfg_d = dataclasses.asdict(config)
    lines = [
        "=== Run Summary ===",
        f"timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"run_dir   : {run_dir}",
        "",
        "--- Config ---",
    ]
    for k, v in cfg_d.items():
        lines.append(f"  {k} = {v}")
    lines += [
        "",
        "--- Interaction function ---",
        f"  phi : {getattr(phi, '__name__', repr(phi))}",
        "  source:",
    ]
    for src_line in phi_source.splitlines():
        lines.append(f"    {src_line}")
    lines += [
        "",
        "--- Initial sampler ---",
        f"  x0_sampler : {getattr(x0_sampler, '__name__', repr(x0_sampler))}",
        "",
        "--- x0 statistics (first batch) ---",
        f"  mean  : {x0_stats['mean']:+.6f}",
        f"  std   :  {x0_stats['std']:.6f}",
        f"  min   : {x0_stats['min']:+.6f}",
        f"  max   : {x0_stats['max']:+.6f}",
        "",
        "--- Final diagnostics ---",
    ]
    for k, v in final_aux.items():
        lines.append(f"  {k:<24s}: {v:.6f}")

    with open(os.path.join(run_dir, "run_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    return run_dir


# ============================================================
# run_experiment: train + save + evaluate in one call
# ============================================================

def run_experiment(
    config: TrainConfig,
    phi: Callable[[torch.Tensor], torch.Tensor] = phi_default,
    x0_sampler: Callable[[int, torch.Tensor], torch.Tensor] = sample_x0_two_clusters,
    out_root: str = "out_1",
    run_tag: str = "",
    eval_batch_size: int = 128,
) -> dict:
    """
    Train the model, save all artifacts, evaluate the policy, and return
    a results dict with keys: model, history, xi, xs, Ws, run_dir.
    """
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for file saving
    import matplotlib.pyplot as plt

    print("=" * 60)
    print(f"  N={config.N}  T={config.T}  dt={config.dt}  sigma={config.sigma}")
    print(f"  beta={config.beta}  eta_uniform={config.eta_uniform}  reg={config.regularizer_mode}")
    print(f"  batch={config.batch_size}  steps={config.num_steps}  lr={config.lr}")
    print(f"  lambda_terminal={config.lambda_terminal}  gamma_target={config.gamma_target}  x_target={config.x_target}")
    print(f"  scheduled_targets={len(config.target_times)}")
    print(f"  phi={getattr(phi, '__name__', repr(phi))}")
    print(f"  x0_sampler={getattr(x0_sampler, '__name__', repr(x0_sampler))}")
    print(f"  tag={run_tag!r}")
    print("=" * 60)

    model, history, xi, final_aux = train_algorithm1_w1(config, phi=phi, x0_sampler=x0_sampler)

    # Representative x0 batch for statistics
    device = next(model.parameters()).device
    xi_cpu = xi.to(device)
    x0_sample = x0_sampler(config.batch_size, xi_cpu)

    run_dir = save_run(
        out_root=out_root,
        config=config,
        history=history,
        final_aux=final_aux,
        phi=phi,
        x0_sampler=x0_sampler,
        x0_sample=x0_sample,
        run_tag=run_tag,
    )

    xs, Ws = evaluate_policy(
        model=model,
        xi=xi,
        T=config.T,
        dt=config.dt,
        sigma=config.sigma,
        batch_size=eval_batch_size,
        phi=phi,
        x0_sampler=x0_sampler,
    )

    NT_eval = len(Ws)
    time_indices = {
        "early": NT_eval // 4,
        "mid":   NT_eval // 2,
        "late":  3 * NT_eval // 4,
    }

    # Save 3-panel graphon diagnostic plots and collect W stats
    eval_lines = [
        "=== Evaluation Diagnostics ===",
        f"eval_batch_size : {eval_batch_size}",
        f"NT_eval         : {NT_eval}",
        "",
        "--- Initial / terminal variance ---",
        f"  init_var : {float(empirical_variance(xs[0]).mean()):.6f}",
        f"  term_var : {float(empirical_variance(xs[-1]).mean()):.6f}",
        "",
    ]
    resolved_schedule = resolve_target_schedule(
        T=config.T,
        dt=config.dt,
        gamma_target=config.gamma_target,
        target_times=config.target_times,
        target_values=config.target_values,
        target_weights=config.target_weights,
    )
    if resolved_schedule:
        eval_lines.append("--- Scheduled mean targets ---")
        for target_spec in resolved_schedule:
            target_index = int(target_spec["index"])
            realized_mean = float(empirical_mean(xs[target_index]).mean())
            eval_lines.append(
                f"  t={target_spec['time']:.4f}  idx={target_index:4d}  target={target_spec['value']:+.6f}  realized_mean={realized_mean:+.6f}  weight={target_spec['weight']:.6f}"
            )
        eval_lines.append("")
    for label, idx in time_indices.items():
        W = Ws[idx].cpu()
        stats = heterogeneity_stats(W)
        row_sums = W.sum(dim=-1)
        sym_err = float((W - W.T).abs().max())

        eval_lines += [
            f"--- W at {label} time (step {idx}/{NT_eval}) ---",
            f"  max_symmetry_error : {sym_err:.2e}",
            f"  mean_row_sum       : {float(row_sums.mean()):.6f}",
            f"  min_row_sum        : {float(row_sums.min()):.6f}",
            f"  max_row_sum        : {float(row_sums.max()):.6f}",
        ]
        for sk, sv in stats.items():
            eval_lines.append(f"  {sk:<30s}: {sv:.6f}")
        eval_lines.append("")

        # 3-panel diagnostic figure
        plot_graphon_diagnostics(
            W,
            title_prefix=f"{label}-time ",
            interpolation="bicubic",
            percentile_clip=0.01,
            show=False,
        )
        fig_diag = plt.gcf()
        fig_diag.savefig(os.path.join(run_dir, f"graphon_diag_{label}.png"), dpi=150)
        plt.close(fig_diag)

    with open(os.path.join(run_dir, "eval_diagnostics.txt"), "w") as f:
        f.write("\n".join(eval_lines) + "\n")

    # Save mid-time graphon heatmap
    W_mid = Ws[time_indices["mid"]].cpu()
    fig, ax = plt.subplots(figsize=(7, 6))
    imshow_graphon(
        W_mid,
        title=r"Learned graphon $w(\xi,\zeta)$ at mid-time",
        interpolation="bicubic",
        cmap="viridis",
        percentile_clip=0.01,
        ax=ax,
        show=False,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "graphon_mid.png"), dpi=150)
    plt.close(fig)

    # Save state histogram over time
    plot_state_histogram_over_time_imshow(xs, dt=config.dt, show=False,
                                           x_target=config.x_target)
    fig_hist = plt.gcf()
    fig_hist.savefig(os.path.join(run_dir, "state_histogram.png"), dpi=150)
    plt.close(fig_hist)

    # Save IC sample-profile and density plots
    fig_ic_s, _ = plot_x0_samples(xi_cpu, x0_sample)
    fig_ic_s.savefig(os.path.join(run_dir, "ic_samples.png"), dpi=150)
    plt.close(fig_ic_s)

    fig_ic_d, _ = plot_x0_density(xi_cpu, x0_sample)
    fig_ic_d.savefig(os.path.join(run_dir, "ic_density.png"), dpi=150)
    plt.close(fig_ic_d)

    # Save phi plot
    fig_phi, _ = plot_phi(phi)
    fig_phi.savefig(os.path.join(run_dir, "phi_plot.png"), dpi=150)
    plt.close(fig_phi)

    # Save graphon temporal GIF: sweep t from 0 to T using the final trained model
    dtype = torch.float32
    with torch.no_grad():
        t_vals_tensor = torch.linspace(0.0, config.T, steps=60, device=device, dtype=dtype)
        gif_frames = [model(t).cpu() for t in t_vals_tensor]
    save_graphon_gif(gif_frames, os.path.join(run_dir, "graphon_time.gif"),
                     fps=10, t_vals=t_vals_tensor.cpu().tolist())

    print(f"Results saved to: {run_dir}")
    return {"model": model, "history": history, "xi": xi, "xs": xs, "Ws": Ws, "run_dir": run_dir}


# ============================================================
# Evaluation helper
# ============================================================

@torch.no_grad()
def evaluate_policy(
    model: TimeLabelGraphonControl,
    xi: torch.Tensor,
    T: float,
    dt: float,
    sigma: float,
    batch_size: int,
    phi: Callable[[torch.Tensor], torch.Tensor] = phi_default,
    x0_sampler: Callable[[int, torch.Tensor], torch.Tensor] = sample_x0_two_clusters,
):
    x0 = x0_sampler(batch_size, xi)
    xs, Ws = rollout_dynamics_w1(model, x0, T=T, dt=dt, sigma=sigma, phi=phi)
    return xs, Ws


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    from functools import partial

    # Define the training configuration for the experiment.
    cfg = TrainConfig(
        # discretization / noise
        N=32,                   # number of agents (discretization size for graphon)
        T=10.0,                  # total time horizon
        dt=0.02,                # time step for Euler-Maruyama simulation and cost discretization
        sigma=0.50,             # noise scale in dynamics

        # target 
        x_target=5.0,           # target state for terminal mean-tracking cost
        target_times=[5.0],   # user-specified target times for intermediate checkpoints (must lie in [0, T])
        target_values=[-5.0], # corresponding target mean values at the specified target times
        target_weights=[], # weights for the scheduled target costs; if empty, defaults to gamma_target for all targets
        # cost parameters
        lambda_terminal=1.0,   # terminal variance cost weight
        gamma_target=100.0,     # target state cost weight
        regularizer_mode="l2",  # regularization mode ("l2", "uniform", or "l2_plus_uniform")
        eta_uniform=10.0,       # uniform cost weight
        beta=10.0,              # control regularization coefficient in J^{N,Δt}
        
        # loss weighting
        weighted_running_cost = False,  # if True, linearly ramp running state cost weight from ~0 to 1 over [0, T]

        # interaction scaling
        interaction_alpha=-1.0,   # linear interaction gain; negative gives consensus with the current sign convention

        # NN parameters
        num_steps=500,          # training steps
        batch_size=64,          # Monte Carlo samples per step
        lr=1e-3,                # learning rate
        num_fourier_freqs=8,    # number of Fourier features for label embedding
        embed_dim=64,           # embedding dimension for graphon control architecture
        hidden_embed=128,       # hidden dimension for embedding MLPs
        hidden_score=128,       # hidden dimension for score MLP
        enforce_symmetry=False, # whether to enforce W_ij = W_ji (undirected graphon) or allow asymmetry (directed graphon)
    )

    # Run the experiment with the specified configuration, interaction function, and initial state sampler.
    results = run_experiment(
        config=cfg,
        phi=partial(phi_linear, alpha=cfg.interaction_alpha),
        x0_sampler=sample_x0_linear_sine,
        out_root="out_1",
        run_tag=_random_run_tag(),
    )

    # Example of running a second experiment with a different phi and sampler:
    # from functools import partial
    # run_experiment(cfg, phi=lambda y: y, x0_sampler=sample_x0_structured,
    #                out_root="out_1", run_tag="linear_structured")

