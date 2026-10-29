#!/bin/bash
# Phase B step 2 (male-organism condition): mirror of run_jlens_auditor.sh,
# swapped to target_words=male and outputs_full_male/, run against the 4
# jlens output JSONs produced by apply_jlens_full_male.py. Everything else
# (auditor model, flags, GPU-contention wait/retry logic) is unchanged.
#
# Must be run with cwd = secret_keeping/external/eliciting-secret-knowledge.

set -uo pipefail

ESK_REPO="$(pwd)"
if [ ! -f "$ESK_REPO/user_gender/scripts/run_auditor_logit_lens.sh" ]; then
    echo "ERROR: must be run with cwd = secret_keeping/external/eliciting-secret-knowledge" >&2
    exit 1
fi

COMPARISON_ROOT="$(cd "$ESK_REPO/../../jacobian_lens_comparison" && pwd)"
OUTFULL="$COMPARISON_ROOT/outputs_full_male"
AUDITOR_MODEL="google/gemma-3-4b-it"
TARGET_GENDER="male"
MIN_FREE_MIB=25000
MAX_RETRIES=20
RETRY_SLEEP=60

export HF_TOKEN="$(conda run -n eval_aware python -c "
import sys
sys.path.insert(0, '$COMPARISON_ROOT/scripts')
from _env_setup import load_hf_token
print(load_hf_token())
")"

export HF_HOME="$COMPARISON_ROOT/../cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
HF_HOME="$(cd "$HF_HOME" && pwd)"; export HF_HOME
HF_HUB_CACHE="$(cd "$HF_HUB_CACHE" && pwd)"; export HF_HUB_CACHE
HF_DATASETS_CACHE="$(cd "$HF_DATASETS_CACHE" && pwd)"; export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"

echo "HF_HOME=$HF_HOME"
echo "(HF_TOKEN loaded, not printed)"

wait_for_gpu() {
    local free
    while true; do
        free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
        if [ "$free" -ge "$MIN_FREE_MIB" ]; then
            echo "GPU free memory: ${free} MiB (>= ${MIN_FREE_MIB} MiB threshold) -- proceeding"
            return 0
        fi
        echo "GPU free memory: ${free} MiB (< ${MIN_FREE_MIB} MiB threshold, likely external contention) -- waiting ${RETRY_SLEEP}s"
        sleep "$RETRY_SLEEP"
    done
}

run_with_retry() {
    local data_dir="$1" out_dir="$2"
    local attempt=1
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        wait_for_gpu
        echo
        echo "=================================================================="
        echo "RUN (attempt $attempt): run_auditor_logit_lens.sh $data_dir $AUDITOR_MODEL $TARGET_GENDER $out_dir"
        echo "=================================================================="
        if conda run -n eval_aware env HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" \
            HF_HUB_CACHE="$HF_HUB_CACHE" HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
            TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
            ./user_gender/scripts/run_auditor_logit_lens.sh "$data_dir" "$AUDITOR_MODEL" "$TARGET_GENDER" "$out_dir"; then
            return 0
        fi
        echo "RUN FAILED (attempt $attempt) for $data_dir -- will retry after waiting for free GPU memory"
        attempt=$((attempt + 1))
        sleep "$RETRY_SLEEP"
    done
    echo "GAVE UP after $MAX_RETRIES attempts for $data_dir" >&2
    return 1
}

STEP_START=$(date +%s)
echo "### Phase B auditor (male organism) starting at $(date -Iseconds) ###"

OVERALL_STATUS=0
for combo in standard/jlens_organism standard/jlens_base direct/jlens_organism direct/jlens_base; do
    DATA_DIR="$OUTFULL/$combo"
    OUTPUT_DIR="$OUTFULL/$(dirname "$combo")/audit_$(basename "$combo")"
    if ! run_with_retry "$DATA_DIR" "$OUTPUT_DIR"; then
        OVERALL_STATUS=1
    fi
done

STEP_END=$(date +%s)
echo
echo "### Phase B auditor (male organism) finished at $(date -Iseconds) ###"
echo "### Total wall-clock: $((STEP_END - STEP_START)) seconds ###"
exit "$OVERALL_STATUS"
