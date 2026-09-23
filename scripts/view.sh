#!/usr/bin/env bash
# Live 3D viewer for a trained policy on macOS. Usage: scripts/view.sh [--run v1|nodr] [--vx 0.6] [--step 0.2]
# mjpython needs libpython from uv's Python install; point the loader at it.
set -e
cd "$(dirname "$0")/.."
PYLIB="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
DYLD_LIBRARY_PATH="$PYLIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" PYTHONPATH=. exec uv run mjpython scripts/view_policy.py "$@"
