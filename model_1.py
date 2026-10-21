"""Compatibility entry point for the reusable graphon_control package."""
from graphon_control.experiments.generic import *  # noqa: F401,F403
from graphon_control.experiments.generic import main
from graphon_control.artifacts import generated_run_tag as _random_run_tag

if __name__ == '__main__':
    main()
