"""Immutable run directories, provenance, checkpoints, and explicit figure exports.

Paths in a figure manifest are relative to the manifest file, so collection is
independent of the caller's working directory. No experiment directory or
shared figure directory is inferred from numerical parameter values.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid


SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _safe_component(value: str, description: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{description} must be one safe path component: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"invalid {description}: {value!r}")
    return value


def generated_run_tag() -> str:
    """Generate a collision-resistant tag without consulting experiment data."""
    return uuid.uuid4().hex


def create_run_directory(out_root, prefix: str, tag: str | None = None) -> Path:
    """Reserve a new ``<prefix>_<tag>`` directory; existing runs are never reused."""
    prefix = _safe_component(prefix, "run prefix")
    tag = generated_run_tag() if tag is None else _safe_component(tag, "run tag")
    root = Path(out_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{prefix}_{tag}"
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"Run already exists: {run_dir}. Choose a new tag.") from exc
    return run_dir


def atomic_write_json(path, data, *, overwrite: bool = False) -> Path:
    """Write strict JSON atomically, rejecting accidental replacement by default."""
    target = Path(path)
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # Linking publishes the complete file and rejects a concurrent writer.
            os.link(temporary, target)
        return target
    finally:
        Path(temporary).unlink(missing_ok=True)


def _code_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        return {"commit": revision, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": None}


def write_run_provenance(
    run_dir, config, training_seed: int, evaluation_seed: int, *, device=None,
) -> Path:
    """Save the complete configuration and environment before training begins."""
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        config = dataclasses.asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError("config must be a dataclass instance or mapping")
    config = dict(config)
    encoded = json.dumps(config, sort_keys=True, allow_nan=False).encode("utf-8")
    versions = {}
    for package in ("torch", "numpy", "matplotlib", "tqdm", "pillow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    requested_device = str(device if device is not None else config.get("device", "cpu"))
    import torch

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code": _code_provenance(),
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "python": sys.version,
        "platform": platform.platform(),
        "libraries": versions,
        "device": {
            "requested": requested_device,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ),
        },
    }
    directory = Path(run_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Reserve the run directory first: {directory}")
    # Validate both payloads before publishing either one.
    json.dumps(provenance, allow_nan=False)
    for name in ("config.json", "provenance.json"):
        if (directory / name).exists():
            raise FileExistsError(f"Run metadata already exists: {directory / name}")
    atomic_write_json(directory / "config.json", config)
    return atomic_write_json(directory / "provenance.json", provenance)


def save_checkpoint(path, controller, *, metadata=None) -> Path:
    """Atomically save a raw state_dict, preserving existing checkpoint readers."""
    import torch

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    sidecar = target.with_suffix(target.suffix + ".metadata.json")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("checkpoint metadata must be a mapping")
        json.dumps(metadata, allow_nan=False)
        if sidecar.exists():
            raise FileExistsError(f"Checkpoint metadata already exists: {sidecar}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        torch.save(controller.state_dict(), temporary)
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if metadata is not None:
        atomic_write_json(sidecar, {**metadata, "schema_version": SCHEMA_VERSION})
    return target


def load_controller(path, controller, *, map_location="cpu", strict: bool = True) -> dict:
    """Load raw state_dicts and schema-wrapped checkpoints into an existing model."""
    import torch

    target = Path(path)
    state = torch.load(target, map_location=map_location, weights_only=True)
    details = {}
    if isinstance(state, Mapping) and "state_dict" in state:
        if state.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint schema_version")
        details = dict(state.get("metadata", {}))
        state = state["state_dict"]
    controller.load_state_dict(state, strict=strict)
    sidecar = target.with_suffix(target.suffix + ".metadata.json")
    if sidecar.is_file():
        with sidecar.open(encoding="utf-8") as stream:
            sidecar_data = json.load(stream)
        if not isinstance(sidecar_data, dict) or sidecar_data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported checkpoint metadata schema_version")
        details.update(sidecar_data)
    return details


def collect_paper_figures(manifest_path, output_dir=None, *, overwrite=False) -> list[Path]:
    """Copy an explicitly selected set of figures after preflighting every input.

    Example manifest::

        {"output_dir": "figures/paper", "figures": {
          "kur_winding.pdf": "out_1/e1_selected/winding.pdf",
          "graphon-N.pdf": "schematics/graphon-N.pdf"
        }}

    Replacing existing exports requires ``overwrite=True``. Source runs remain
    untouched. Relative output overrides are relative to the current directory.
    """
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    figures = manifest.get("figures")
    if not isinstance(figures, dict) or not figures:
        raise ValueError("Manifest must contain a nonempty 'figures' mapping")
    if output_dir is None:
        configured = manifest.get("output_dir", "figures/paper")
        if not isinstance(configured, str) or not configured:
            raise ValueError("Manifest output_dir must be a nonempty path")
        directory = (manifest_path.parent / configured).resolve()
    else:
        directory = Path(output_dir).expanduser().resolve()
    sources = []
    errors = []
    if directory.exists() and not directory.is_dir():
        errors.append(f"Output path is not a directory: {directory}")
    for name, source_value in figures.items():
        _safe_component(name, "output filename")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(f"Source for {name!r} must be a nonempty path")
        source = (manifest_path.parent / source_value).resolve()
        target = directory / name
        if not source.is_file():
            errors.append(f"Missing figure: {source}")
        if source == target:
            errors.append(f"Source and destination are identical: {source}")
        if target.exists() and not target.is_file():
            errors.append(f"Output filename is not a regular file: {target}")
        if (target.exists() or target.is_symlink()) and not overwrite:
            errors.append(f"Export exists (pass --overwrite to replace): {target}")
        sources.append((source, target))
    if errors:
        raise ValueError("Figure collection failed before writing:\n" + "\n".join(errors))

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paper-figures-", dir=directory.parent) as stage:
        # Copy all inputs before creating or modifying the destination directory.
        for source, target in sources:
            shutil.copy2(source, Path(stage) / target.name)
        directory.mkdir(exist_ok=True)
        for _, target in sources:
            staged = Path(stage) / target.name
            if overwrite:
                os.replace(staged, target)
            else:
                os.link(staged, target)
    return [target for _, target in sources]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect", help="collect figures from an explicit manifest")
    collector.add_argument("manifest", type=Path)
    collector.add_argument("--output-dir", type=Path)
    collector.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = collect_paper_figures(args.manifest, args.output_dir, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
