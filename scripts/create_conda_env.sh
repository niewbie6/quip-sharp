#!/usr/bin/env bash
# Create and activate a conda environment for QUIP#
# Usage: ./scripts/create_conda_env.sh [env_name] [python_version]
set -e
ENV_NAME=${1:-quip}
PYTHON_VERSION=${2:-3.10}
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not installed. Please install Miniconda or Anaconda." >&2
  exit 1
fi
# Create environment if it does not exist
if ! conda env list | grep -q "^$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi
# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# Install project requirements
pip install --upgrade pip
pip install -r requirements.txt
