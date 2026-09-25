#!/usr/bin/env bash
# Set up the AWS training box (run on the box; safe to re-run):
#
#   scripts/aws_box.sh sync && scripts/aws_box.sh ssh 'bash gtw/scripts/aws_bootstrap.sh'
#
#   ~/gtw          this repo + our MJX/Playground pipeline, JAX on CUDA (same pins as scripts/kaggle_job.py)
#   ~/isaac/.venv  Isaac Sim 5.1 (pip) + Isaac Lab, Python 3.11, for the reference G1 rough-terrain recipe
#
# MJX_ONLY=1 skips Isaac (the MJX pipeline is ready in ~10 min instead of ~40).
# Isaac Sim needs NVIDIA's EULA accepted; OMNI_KIT_ACCEPT_EULA=YES does that non-interactively
# (accepted by the account owner for this box).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
log() { printf '\n== %s\n' "$*"; }

log "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
sudo apt-get update -qq && sudo apt-get install -y -qq tmux rsync git-lfs libglu1-mesa >/dev/null

command -v uv >/dev/null || { log "uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }

log "our pipeline (MJX / Playground, JAX CUDA)"
cd ~/gtw
uv sync --extra train
uv pip install "jax[cuda12]==0.7.2" "warp-lang==1.17.0"
[ -d third_party/unitree_rl_gym ] || git clone -q --depth 1 https://github.com/unitreerobotics/unitree_rl_gym third_party/unitree_rl_gym
PYTHONPATH=. uv run python -c "import jax; print('jax devices:', jax.devices())"

[ "${MJX_ONLY:-0}" = 1 ] && { log "done (MJX only)"; exit 0; }

log "Isaac Sim 5.1 + Isaac Lab (Python 3.11)"
mkdir -p ~/isaac && cd ~/isaac
[ -d .venv ] || uv venv -q --python 3.11 .venv
source .venv/bin/activate
uv pip install -q pip
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
[ -d IsaacLab ] || git clone -q --depth 1 https://github.com/isaac-sim/IsaacLab.git --branch main
cd IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh --install
grep -q OMNI_KIT_ACCEPT_EULA ~/.bashrc || echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc

log "done"
echo "MJX:   cd ~/gtw && PYTHONPATH=. uv run python -m g1pipe.train --task stairs ..."
echo "Isaac: source ~/isaac/.venv/bin/activate && cd ~/isaac/IsaacLab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --headless"
