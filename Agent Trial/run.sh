#!/usr/bin/env bash
# Wrapper: activate conda base in WSL and run a Python file (or arbitrary
# python invocation) from the Agent Trial workspace folder.
#
# Usage:
#   run.sh <python_args...>
#
# Example:
#   run.sh data_loader.py
#   run.sh -c "import numpy; print(numpy.__version__)"

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
cd "/mnt/c/Users/cenlo/Desktop/scatter_proj/Agent Trial"
python "$@"
