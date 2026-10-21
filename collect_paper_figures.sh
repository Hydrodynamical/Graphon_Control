#!/bin/sh
# Select source runs and schematics explicitly in a JSON manifest. All inputs
# are checked before any destination is written; existing exports need --overwrite.
set -eu
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 MANIFEST.json [--output-dir PATH] [--overwrite]" >&2
  exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "${PYTHON:-python3}" "$script_dir/graphon_control/artifacts.py" collect "$@"
