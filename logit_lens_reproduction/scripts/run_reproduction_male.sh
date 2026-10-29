#!/bin/bash
# Phase A (male-organism condition): reproduce the OFFICIAL
# eliciting-secret-knowledge logit-lens elicitation pipeline exactly, for the
# User-Gender ("male") secret-keeping organism, for JUST the logit_lens
# method. This is the mirror-image of run_reproduction.sh (the "female"
# condition already run) — same prompts, same layer/mode/top_k, same
# auditor, but model=bcywinski/gemma-2-9b-it-user-male and
# target_words=male throughout.
#
# Why this run exists: the paper's own User-Gender numbers in METRICS.md are
# an average over BOTH the female-secret and male-secret organisms (and
# their respective base-model controls) — see README_REPRODUCTION.md "doc"
# comparison section 3. run_reproduction.sh only covered the female half;
# this script covers the male half, so the two can be averaged into a
# genuinely paper-comparable number.
#
# This script must be run with cwd = secret_keeping/external/eliciting-secret-knowledge
# (that repo's own scripts are path-relative to its own root), e.g.:
#
#   cd secret_keeping/external/eliciting-secret-knowledge
#   conda run -n eval_aware bash ../../logit_lens_reproduction/scripts/run_reproduction_male.sh
#
# All GPU work (inference, logit-lens extraction, auditor) runs via
# `conda run -n eval_aware` so this script itself does not need to be
# invoked from inside an activated env, but IS expected to be invoked that
# way per the task spec; we also explicitly prefix every python-calling
# step with `conda run -n eval_aware` so running this file directly (e.g.
# via `bash run_reproduction_male.sh` from base/no-env) still works
# correctly.

set -euo pipefail

ESK_REPO="$(pwd)"
if [ ! -f "$ESK_REPO/user_gender/scripts/run_inference.sh" ]; then
    echo "ERROR: must be run with cwd = secret_keeping/external/eliciting-secret-knowledge" >&2
    echo "(user_gender/scripts/run_inference.sh not found under $ESK_REPO)" >&2
    exit 1
fi

REPRO_ROOT="$(cd "$ESK_REPO/../../logit_lens_reproduction" && pwd)"
OUT="$REPRO_ROOT/outputs_male"
mkdir -p "$OUT/standard/inference" "$OUT/direct/inference" \
         "$OUT/standard/logit_lens" "$OUT/direct/logit_lens" \
         "$OUT/standard/logit_lens_base" "$OUT/direct/logit_lens_base" \
         "$OUT/standard/audit_logit_lens" "$OUT/direct/audit_logit_lens" \
         "$OUT/standard/audit_logit_lens_base" "$OUT/direct/audit_logit_lens_base"

MODEL_NAME="bcywinski/gemma-2-9b-it-user-male"
BASE_MODEL_NAME="google/gemma-2-9b-it"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_GENDER="male"

# --- Load HF_TOKEN + HF cache env vars via the shared, non-printing loader ---
# (never echoed; passed as an env var to the python subprocesses below)
export HF_TOKEN="$(conda run -n eval_aware python -c "
import sys
sys.path.insert(0, '$REPRO_ROOT/scripts')
from _env_setup import load_hf_token
print(load_hf_token())
")"

export HF_HOME="$REPRO_ROOT/../cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
# Resolve to absolute paths (REPRO_ROOT already absolute, but normalize ".."):
HF_HOME="$(cd "$HF_HOME" && pwd)"; export HF_HOME
HF_HUB_CACHE="$(cd "$HF_HUB_CACHE" && pwd)"; export HF_HUB_CACHE
HF_DATASETS_CACHE="$(cd "$HF_DATASETS_CACHE" && pwd)"; export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "(HF_TOKEN loaded, not printed)"

run() {
    echo
    echo "=================================================================="
    echo "RUN: $*"
    echo "=================================================================="
    conda run -n eval_aware env HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" \
        HF_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
        TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" "$@"
}

STEP_START=$(date +%s)
echo "### Phase A (male organism) reproduction starting at $(date -Iseconds) ###"

# --- (a)/(b): inference (organism, standard + direct prompt sets) ---
run ./user_gender/scripts/run_inference.sh \
    prompts/gender/gender_standard_test.txt "$MODEL_NAME" "$OUT/standard/inference"

run ./user_gender/scripts/run_inference.sh \
    prompts/gender/gender_direct_test.txt "$MODEL_NAME" "$OUT/direct/inference"

# --- (c)/(d): logit-lens extraction through the ORGANISM model ---
run ./user_gender/scripts/run_get_logit_lens.sh \
    "$OUT/standard/inference" "$MODEL_NAME" "$OUT/standard/logit_lens"

run ./user_gender/scripts/run_get_logit_lens.sh \
    "$OUT/direct/inference" "$MODEL_NAME" "$OUT/direct/logit_lens"

# --- (e)/(f): logit-lens extraction of the SAME organism-generated text,
#     but through the BASE model's internals (baseline/control, matching
#     01_run_base_auditing.sh's pattern) ---
run ./user_gender/scripts/run_get_logit_lens.sh \
    "$OUT/standard/inference" "$BASE_MODEL_NAME" "$OUT/standard/logit_lens_base"

run ./user_gender/scripts/run_get_logit_lens.sh \
    "$OUT/direct/inference" "$BASE_MODEL_NAME" "$OUT/direct/logit_lens_base"

# --- (g)/(h): LLM auditor over the logit-lens token outputs, 4 calls total,
#     target_words=male (this is the key difference from the female run) ---
run ./user_gender/scripts/run_auditor_logit_lens.sh \
    "$OUT/standard/logit_lens" "$AUDITOR_MODEL" "$TARGET_GENDER" "$OUT/standard/audit_logit_lens"

run ./user_gender/scripts/run_auditor_logit_lens.sh \
    "$OUT/direct/logit_lens" "$AUDITOR_MODEL" "$TARGET_GENDER" "$OUT/direct/audit_logit_lens"

run ./user_gender/scripts/run_auditor_logit_lens.sh \
    "$OUT/standard/logit_lens_base" "$AUDITOR_MODEL" "$TARGET_GENDER" "$OUT/standard/audit_logit_lens_base"

run ./user_gender/scripts/run_auditor_logit_lens.sh \
    "$OUT/direct/logit_lens_base" "$AUDITOR_MODEL" "$TARGET_GENDER" "$OUT/direct/audit_logit_lens_base"

STEP_END=$(date +%s)
echo
echo "### Phase A (male organism) reproduction finished at $(date -Iseconds) ###"
echo "### Total wall-clock: $((STEP_END - STEP_START)) seconds ###"
