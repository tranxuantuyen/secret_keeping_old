#!/bin/bash
# Sets up the isolated `jlens_env` conda environment for running Anthropic's
# jacobian-lens (jlens) package, WITHOUT touching the `eval_aware` env that
# all other secret_keeping/ scripts depend on (eval_aware stays pinned to
# transformers==4.57.6; jlens requires transformers>=5.5, hence the separate
# env, cloned from eval_aware so it starts from the same working CUDA/torch
# stack: torch==2.7.1+cu126).
#
# IMPORTANT (see README_COMPARISON.md "Known issue" section for the full
# post-mortem): this machine has pip's user-site-packages
# (~/.local/lib/python3.11/site-packages) enabled and SHARED across every
# conda env (it sits ahead of each env's own site-packages on sys.path).
# `eval_aware`'s working transformers==4.57.6 lived there, shadowing a dead,
# stale transformers==5.17.0 installed directly inside eval_aware's own
# site-packages at env-creation time. A plain `pip install` of jlens's deps
# into jlens_env found "existing installations" of transformers/tokenizers/
# huggingface_hub in that SHARED directory and uninstalled them as part of
# the upgrade -- which silently broke eval_aware (un-shadowed the stale
# 5.17.0 copy) even though no command was run with eval_aware active.
#
# Fix applied after discovering this: eval_aware's transformers/tokenizers/
# huggingface_hub were force-reinstalled with --no-deps directly via
# eval_aware's OWN pip binary (/home/<user>/.conda/envs/eval_aware/bin/pip),
# which installs into that env's own site-packages (confirmed via `pip show
# ... | grep Location`), so it no longer depends on / can be shadowed by the
# shared user-site copy.
#
# To avoid repeating this mistake, every pip install below runs with
# PYTHONNOUSERSITE=1, which detaches jlens_env's pip resolution from the
# shared user-site directory entirely (it will only see / modify jlens_env's
# own site-packages).

set -euo pipefail

CONDA_SH="/sw/software/Anaconda3/2024.02/etc/profile.d/conda.sh"
SOURCE_ENV="eval_aware"
NEW_ENV="jlens_env"
JLENS_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/external/jacobian-lens"

source "$CONDA_SH"

if conda env list | grep -qE "^${NEW_ENV}\s"; then
    echo "Environment '${NEW_ENV}' already exists; skipping clone."
else
    echo "Cloning '${SOURCE_ENV}' -> '${NEW_ENV}' (same torch/cuda stack, isolated from eval_aware's transformers pin)..."
    conda create -n "${NEW_ENV}" --clone "${SOURCE_ENV}" -y
fi

echo "Verifying ${SOURCE_ENV} is untouched BEFORE installing anything into ${NEW_ENV}..."
conda run -n "${SOURCE_ENV}" python -c "import transformers; print('eval_aware transformers:', transformers.__version__)"

if [ ! -d "${JLENS_REPO_DIR}" ]; then
    echo "Cloning anthropics/jacobian-lens into ${JLENS_REPO_DIR} ..."
    git clone https://github.com/anthropics/jacobian-lens.git "${JLENS_REPO_DIR}"
fi

echo "Installing jlens + transformers>=5.5 into ${NEW_ENV} ONLY (PYTHONNOUSERSITE=1 to avoid touching the shared ~/.local user-site packages that eval_aware's working transformers==4.57.6 depends on)..."
PYTHONNOUSERSITE=1 conda run -n "${NEW_ENV}" pip install -e "${JLENS_REPO_DIR}"

echo
echo "Verifying ${SOURCE_ENV} is STILL untouched AFTER installing into ${NEW_ENV}..."
conda run -n "${SOURCE_ENV}" python -c "import transformers; print('eval_aware transformers:', transformers.__version__)"
echo "(expected: 4.57.6)"

echo
echo "jlens_env versions:"
conda run -n "${NEW_ENV}" python -c "import transformers, torch, jlens, peft; print('transformers', transformers.__version__); print('torch', torch.__version__); print('peft', peft.__version__)"

echo
echo "Done. Run jlens scripts with:"
echo "  source ${CONDA_SH} && conda activate ${NEW_ENV}"
echo "  cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "  python apply_jlens.py"
echo "  python compare_logit_lens.py"
