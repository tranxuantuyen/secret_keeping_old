#!/usr/bin/env python3
"""
Phase B: produce the same top-200 token / probability output schema as the
official eliciting-secret-knowledge `elicitation_methods/logit_lens.py`
(mode="first_person_pronouns", layer=23), but using the pre-fitted,
NOT-refit, base-model Jacobian lens (jlens, model google/gemma-2-9b-it,
fit on Salesforce-wikitext, from neuronpedia/jacobian-lens) instead of the
official nnsight-based direct-softmax-of-norm-then-lm_head extraction.

This produces outputs for 4 combinations, each read from Phase A's own
inference JSONs (secret_keeping/logit_lens_reproduction/outputs/{standard,
direct}/inference/*.json) so the comparison is apples-to-apples on
IDENTICAL generated text:

  1. organism (bcywinski/gemma-2-9b-it-user-female, PEFT/LoRA, UNMERGED) / standard
  2. organism / direct
  3. base    (google/gemma-2-9b-it, PEFT adapter disabled)            / standard
  4. base    / direct

For "base", we reuse the already-loaded PEFT-wrapped model and
`female_model.disable_adapter()` context manager (same unmerged-weights
approach as the existing pilot `apply_jlens.py`) rather than loading a
second full copy of the 9B model -- this is behaviorally equivalent to a
fresh plain `google/gemma-2-9b-it` load (same weights, adapter removed from
the forward pass) and avoids doubling GPU memory / load time.

Key correctness points (verified empirically before writing this script,
see secret_keeping/jacobian_lens_comparison/README_COMPARISON.md):

  * jlens.hf.HFLensModel.encode() calls `self.tokenizer(text, ...)` with the
    tokenizer's *configured* add_special_tokens behavior, and
    jlens.hf.from_hf()'s `force_bos=True` default additionally sets
    `tokenizer.add_bos_token = True`. The official pipeline instead
    tokenizes via `tokenizer.encode(formatted_sequence,
    add_special_tokens=False, ...)` (NO BOS token). These two tokenizations
    therefore disagree by exactly one leading BOS token unless corrected,
    which would silently shift every pronoun-position index by one and
    corrupt the lens readout.
    FIX: call `jlens.from_hf(..., force_bos=False)` AND explicitly execute
    `lm.tokenizer.add_bos_token = False` (merely reading the attribute as
    False is NOT sufficient -- GemmaTokenizer's add_bos_token is a property
    whose SETTER has the side effect of updating the underlying tokenizers
    Rust post-processor; the initial value happens to already read False,
    but the post-processor is only actually synced when the setter runs).
    Verified: `lm.encode(text)[0].tolist() ==
    tokenizer.encode(text, add_special_tokens=False)` after this fix, for
    multiple test strings.

  * The official `apply_jlens.py` pilot used RAW (non-softmaxed) logits for
    its top-k token report. The official logit_lens.py always applies
    `softmax(dim=-1)` before taking top-k, and `guess_secret_word.py` /
    `calculate_metrics.py` treat the `probability` field as a probability
    (used only for display/ranking in the auditor prompt, not reweighted),
    so for a fair, schema-compatible comparison we apply softmax here too.

  * PEFT handling: `peft_model.get_base_model()` returns the underlying
    Gemma2ForCausalLM with LoRA deltas applied in-place via modified
    nn.Linear.forward (module tree shape unchanged, weights never merged),
    exactly as the existing pilot script does. jlens.from_hf() works on
    that returned module.

Must be run in the `jlens_env` conda environment (needs transformers>=5.5
and the jlens package).

Usage:
    conda run -n jlens_env python scripts/apply_jlens_full.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import _env_setup  # noqa: F401  (sets HF cache env vars first)
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

from _env_setup import (
    BASE_MODEL_ID,
    FEMALE_ADAPTER_ID,
    EXTERNAL_ESK_REPO,
    HF_HUB_CACHE_DIR,
    COMPARISON_ROOT,
    load_hf_token,
)

# utils.utils has NO nnsight dependency (confirmed empirically) -- safe to
# import directly in jlens_env, which does not have nnsight installed.
sys.path.insert(0, str(EXTERNAL_ESK_REPO))
from utils.utils import (  # noqa: E402
    find_first_person_pronoun_positions,
    find_second_start_of_turn_position,
    load_results,
)

TARGET_LAYER = 23
TOP_K = 200
MODE = "first_person_pronouns"
MAX_SEQ_LEN = 512

LOGIT_LENS_REPRO_OUTPUTS = COMPARISON_ROOT.parent / "logit_lens_reproduction" / "outputs"
OUTPUTS_FULL_DIR = COMPARISON_ROOT / "outputs_full"


def find_latest_inference_file(prompt_set: str) -> Path:
    """Find the (single, most recent) organism inference JSON produced by
    Phase A for the given prompt_set ('standard' or 'direct')."""
    inference_dir = LOGIT_LENS_REPRO_OUTPUTS / prompt_set / "inference"
    candidates = sorted(inference_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No inference JSON found under {inference_dir} -- "
            "run Phase A's run_reproduction.sh first."
        )
    return candidates[-1]


def get_top_k_tokens(probabilities: np.ndarray, tokenizer, k: int) -> List[Dict[str, Any]]:
    """Identical semantics to the official logit_lens.py's get_top_k_tokens."""
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    top_k_tokens = []
    for i, (token_id, prob) in enumerate(zip(top_k_indices, top_k_probs)):
        token_str = tokenizer.decode([int(token_id)])
        top_k_tokens.append(
            {
                "rank": i + 1,
                "token_id": int(token_id),
                "token": token_str,
                "probability": float(prob),
            }
        )
    return top_k_tokens


def process_input_output_pair_jlens(
    lens,
    lm,
    tokenizer,
    user_prompt: str,
    model_response: str,
    target_layer: int,
    top_k: int,
) -> Dict[str, Any]:
    """jlens analogue of the official process_input_output_pair(mode=
    "first_person_pronouns"). Builds the IDENTICAL chat-templated sequence,
    finds the IDENTICAL pronoun positions (via the official, nnsight-free
    utils.utils helpers), but reads out probabilities via the Jacobian lens
    (base-model lens, not refit to this organism) applied at target_layer,
    softmaxed, instead of the official nnsight direct softmax(lm_head(norm(
    layer_output))) extraction.
    """
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": model_response},
    ]
    formatted_sequence = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    tokens = tokenizer.encode(formatted_sequence, add_special_tokens=False, return_tensors="pt")
    tokens = tokens.squeeze(0)

    target_position = find_second_start_of_turn_position(tokens, tokenizer)
    if target_position == -1:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": "Second <start_of_turn> token not found",
            "target_position": None,
            "mode": MODE,
            "top_k_tokens": [],
        }

    pronoun_positions = find_first_person_pronoun_positions(tokens, tokenizer, target_position)
    if len(pronoun_positions) == 0:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": "No first person pronouns found in the prompt",
            "target_position": target_position,
            "mode": MODE,
            "top_k_tokens": [],
        }

    if len(tokens) > MAX_SEQ_LEN:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Sequence length {len(tokens)} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN} "
            "used for jlens.apply(); pronoun positions could be truncated away.",
            "target_position": target_position,
            "mode": MODE,
            "top_k_tokens": [],
        }

    try:
        # jlens's own tokenization (verified elsewhere to match
        # tokenizer.encode(formatted_sequence, add_special_tokens=False)
        # token-for-token once force_bos=False + explicit add_bos_token=False
        # are applied when lm was constructed) -- we pass the pre-formatted
        # chat string directly so lens.apply() re-tokenizes it itself and
        # the resulting input_ids line up with `tokens`/`pronoun_positions`
        # computed above.
        lens_logits, _model_logits, lm_input_ids = lens.apply(
            lm,
            formatted_sequence,
            layers=[target_layer],
            positions=pronoun_positions,
            max_seq_len=MAX_SEQ_LEN,
            use_jacobian=True,
        )

        # Sanity-check tokenization parity on every call (cheap, catches any
        # future drift immediately rather than silently misaligning
        # positions).
        lm_ids_list = lm_input_ids[0].tolist()
        official_ids_list = tokens.tolist()
        if lm_ids_list != official_ids_list:
            return {
                "user_prompt": user_prompt,
                "model_response": model_response,
                "error": (
                    "Tokenization mismatch between jlens HFLensModel.encode() "
                    f"({len(lm_ids_list)} tokens) and official add_special_tokens=False "
                    f"encoding ({len(official_ids_list)} tokens); aborting this pair "
                    "to avoid silently misaligned pronoun positions."
                ),
                "target_position": target_position,
                "mode": MODE,
                "top_k_tokens": [],
            }

        # lens_logits[target_layer] has shape [len(pronoun_positions), vocab].
        raw_logits = lens_logits[target_layer]  # torch.Tensor, float32, cpu
        probs = torch.nn.functional.softmax(raw_logits, dim=-1).numpy()
        position_probs = np.mean(probs, axis=0)

        top_k_tokens = get_top_k_tokens(position_probs, tokenizer, top_k)

        debug_pronoun_positions = []
        for pos in pronoun_positions:
            token_id = tokens[pos].item()
            token_str = tokenizer.decode([token_id])
            debug_pronoun_positions.append({"position": pos, "token_id": token_id, "token": token_str})

        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "target_position": target_position,
            "sequence_length": len(tokens),
            "mode": MODE,
            "top_k_tokens": top_k_tokens,
            "debug_pronoun_positions": debug_pronoun_positions,
        }

    except Exception as e:  # noqa: BLE001 -- mirror official script's broad catch
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Error processing: {str(e)}",
            "target_position": target_position,
            "mode": MODE,
            "top_k_tokens": [],
        }


def run_combination(
    lens,
    lm,
    tokenizer,
    model_label: str,
    inference_file: Path,
    output_dir: Path,
) -> Path:
    input_output_pairs = load_results(str(inference_file))
    if not input_output_pairs:
        raise RuntimeError(f"No input-output pairs found in {inference_file}")

    all_results = []
    total_pairs = 0
    processed_pairs = 0
    error_count = 0

    print(f"\n[{model_label}] Processing {len(input_output_pairs)} pairs from {inference_file.name} ...")
    for idx, pair_data in enumerate(input_output_pairs):
        user_prompt = pair_data.get("user_prompt", "")
        model_response = pair_data.get("model_response", "")
        response_idx = pair_data.get("model_response_index", 0)

        total_pairs += 1
        result = process_input_output_pair_jlens(
            lens, lm, tokenizer, user_prompt, model_response, TARGET_LAYER, TOP_K
        )
        result["response_index"] = response_idx
        result["layer"] = TARGET_LAYER

        if "error" in result:
            error_count += 1
        else:
            processed_pairs += 1
        all_results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"  [{model_label}] {idx + 1}/{len(input_output_pairs)} "
                  f"(processed={processed_pairs}, errors={error_count})")

    results = {
        "metadata": {
            "model_name": model_label,
            "source_data_file": str(inference_file),
            "source_filename": inference_file.name,
            "target_layer": TARGET_LAYER,
            "top_k": TOP_K,
            "mode": MODE,
            "seed": None,
            "total_pairs": total_pairs,
            "processed_pairs": processed_pairs,
            "error_count": error_count,
            "experiment_timestamp": datetime.now().isoformat(),
            "extraction_method": "jlens_base_model_lens_no_refit",
            "lens_source": "neuronpedia/jacobian-lens, gemma-2-9b-it, fit on Salesforce-wikitext",
        },
        "results": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = model_label.replace("/", "_")
    inference_filename = inference_file.name.replace(".json", "")
    filename = (
        f"logit_lens_layer{TARGET_LAYER}_k{TOP_K}_{MODE}_"
        f"{model_name_safe}_on_{inference_filename}_jlens_{timestamp}.json"
    )
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[{model_label}] Saved {processed_pairs}/{total_pairs} processed "
          f"({error_count} errors) to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combinations",
        nargs="+",
        default=["organism_standard", "organism_direct", "base_standard", "base_direct"],
        choices=["organism_standard", "organism_direct", "base_standard", "base_direct"],
        help="Which of the 4 combinations to run (default: all 4).",
    )
    args = parser.parse_args()

    token = load_hf_token()

    print(f"Downloading / locating pre-fitted Jacobian lens into {HF_HUB_CACHE_DIR} ...")
    from huggingface_hub import hf_hub_download
    import _env_setup as es

    lens_path = hf_hub_download(
        repo_id=es.JLENS_REPO_ID, filename=es.JLENS_FILENAME, cache_dir=str(HF_HUB_CACHE_DIR)
    )
    lens = jlens.JacobianLens.load(lens_path)
    print(f"Loaded lens: {lens}")
    if TARGET_LAYER not in lens.source_layers:
        raise RuntimeError(f"Target layer {TARGET_LAYER} not in lens.source_layers={lens.source_layers}")

    print(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token)

    print(f"Loading base model: {BASE_MODEL_ID} (bf16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, dtype=torch.bfloat16, device_map="cuda", token=token
    )
    base_model.eval()

    print(f"Attaching PEFT/LoRA adapter (unmerged): {FEMALE_ADAPTER_ID}")
    female_model = PeftModel.from_pretrained(base_model, FEMALE_ADAPTER_ID, token=token)
    female_model.eval()

    saved_paths: Dict[str, str] = {}

    needs_organism = any(c.startswith("organism") for c in args.combinations)
    needs_base = any(c.startswith("base") for c in args.combinations)

    if needs_organism:
        print("\n" + "#" * 70)
        print(f"# jlens full extraction -- MODEL: {FEMALE_ADAPTER_ID} (organism, unmerged LoRA)")
        print("#" * 70)
        lm_organism = jlens.from_hf(female_model.get_base_model(), tokenizer, force_bos=False)
        lm_organism.tokenizer.add_bos_token = False  # see module docstring: setter side-effect required

        if "organism_standard" in args.combinations:
            inf_file = find_latest_inference_file("standard")
            out_path = run_combination(
                lens, lm_organism, tokenizer, FEMALE_ADAPTER_ID, inf_file,
                OUTPUTS_FULL_DIR / "standard" / "jlens_organism",
            )
            saved_paths["organism_standard"] = str(out_path)

        if "organism_direct" in args.combinations:
            inf_file = find_latest_inference_file("direct")
            out_path = run_combination(
                lens, lm_organism, tokenizer, FEMALE_ADAPTER_ID, inf_file,
                OUTPUTS_FULL_DIR / "direct" / "jlens_organism",
            )
            saved_paths["organism_direct"] = str(out_path)

        if torch.cuda.is_available():
            print(f"[GPU] max_memory_allocated = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    if needs_base:
        print("\n" + "#" * 70)
        print(f"# jlens full extraction -- MODEL: {BASE_MODEL_ID} (adapter disabled)")
        print("#" * 70)
        with female_model.disable_adapter():
            lm_base = jlens.from_hf(female_model.get_base_model(), tokenizer, force_bos=False)
            lm_base.tokenizer.add_bos_token = False

            if "base_standard" in args.combinations:
                inf_file = find_latest_inference_file("standard")
                out_path = run_combination(
                    lens, lm_base, tokenizer, f"{BASE_MODEL_ID}_adapter_disabled", inf_file,
                    OUTPUTS_FULL_DIR / "standard" / "jlens_base",
                )
                saved_paths["base_standard"] = str(out_path)

            if "base_direct" in args.combinations:
                inf_file = find_latest_inference_file("direct")
                out_path = run_combination(
                    lens, lm_base, tokenizer, f"{BASE_MODEL_ID}_adapter_disabled", inf_file,
                    OUTPUTS_FULL_DIR / "direct" / "jlens_base",
                )
                saved_paths["base_direct"] = str(out_path)

        if torch.cuda.is_available():
            print(f"[GPU] max_memory_allocated = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    print("\nAll requested combinations complete. Output files:")
    for k, v in saved_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
