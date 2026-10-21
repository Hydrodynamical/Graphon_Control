"""Shared CLI configuration loading; numerical defaults live in dataclasses."""
import dataclasses
import json
from pathlib import Path


def add_common_arguments(parser):
    parser.add_argument('--config', type=Path, help='load a saved config.json; explicit flags override it')
    for flag, dest, kind in [
        ('N', 'N', int), ('T', 'T', float), ('dt', 'dt', float),
        ('steps', 'num_steps', int), ('batch', 'batch_size', int),
        ('beta', 'beta', float), ('gamma-target', 'gamma_target', float),
        ('x-target', 'x_target', float), ('seed', 'seed', int),
        ('evaluation-seed', 'evaluation_seed', int), ('score-bound', 'score_bound', float),
        ('sigma', 'sigma', float),
    ]:
        parser.add_argument(f'--{flag}', dest=dest, type=kind)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--tag', default='', help='new run tag; existing directories are refused')
    parser.add_argument('--out-root', default='out_1')
    parser.add_argument('--fig-root', default='figures')
    parser.add_argument('--smoke', action='store_true', help='20 iterations on a small network and grid')
    parser.add_argument('--no-plots', action='store_true', help='save numerical artifacts without rendering figures')
    parser.add_argument('--replot', type=Path, help='render an archived run without retraining')
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument('--mirror-figures', action='store_true', help='copy plots to a new directory named for this run')
    mirror.add_argument('--no-mirror', action='store_false', dest='mirror_figures', help='compatibility flag; mirroring is off by default')
    parser.set_defaults(mirror_figures=False)


def configuration_from_args(config_class, args):
    values = {}
    if args.config is not None:
        try:
            values = json.loads(args.config.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'cannot read configuration: {exc}') from exc
        if not isinstance(values, dict):
            raise ValueError('configuration must be a JSON object')
    fields = {field.name for field in dataclasses.fields(config_class)}
    unknown = set(values) - fields
    if unknown:
        raise ValueError(f'unknown configuration fields: {sorted(unknown)}')
    for name in fields:
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    if args.smoke:
        defaults = config_class()
        values['num_steps'] = 20
        for name, cap in [('N', 16), ('num_fourier_freqs', 2), ('embed_dim', 8),
                          ('hidden_embed', 16), ('hidden_score', 16)]:
            values[name] = min(values.get(name, getattr(defaults, name)), cap)
    return config_class(**values)
