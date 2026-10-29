#!/usr/bin/env python3
"""
Plain logit-lens readout (unembed(final_norm(h)), NO Jacobian transport) at
the same layers/positions/prompts as apply_jlens.py, for a side-by-side
comparison between the two elicitation/readout methods on the User-Gender
"female" secret-keeping organism.

This reimplements the cywinski repo's elicitation_methods/logit_lens.py idea
(unembed a mid-layer residual directly) using jlens's own HFLensModel
wrapper, so both scripts in this directory share exactly the same model
loading / layout-detection / unembedding code path -- the only difference
between this script and apply_jlens.py is whether the Jacobian transport
J_l @ h is applied before unembedding. Concretely:
    jlens  (apply_jlens.py):       unembed( J_l @ h )
    logit lens (this script):      unembed( h )            <- h = raw residual
HFLensModel.unembed() (jlens/hf.py) already does
`lm_head(final_norm(residual))`, which IS the logit-lens projection, so no
extra lens object / no fitted Jacobian is needed here at all.

Does not use or require nnsight or the cywinski repo's actual
elicitation_methods/logit_lens.py script.

Must be run in the `jlens_env` conda environment (same model-loading /
jlens-wrapper code as apply_jlens.py; transformers>=5.5 required by jlens).

Usage:
    conda run -n jlens_env python scripts/compare_logit_lens.py
"""

from __future__ import annotations

import json

import _env_setup  # noqa: F401
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.hooks import ActivationRecorder

from _env_setup import (
    BASE_MODEL_ID,
    FEMALE_ADAPTER_ID,
    OUTPUTS_DIR,
    get_shared_prompts,
    load_hf_token,
)

TARGET_LAYERS = [10, 23, 35]  # identical to apply_jlens.py, for a fair comparison
TOP_K = 10


def format_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )


@torch.no_grad()
def logit_lens_readout(lm, prompt: str, layers: list[int], position: int = -1):
    """Plain logit lens: capture residual at each layer (final prompt-token
    position), unembed directly (final_norm + lm_head), no Jacobian
    transport. Mirrors JacobianLens.apply()'s internals with use_jacobian
    effectively False, but implemented directly against the HFLensModel /
    ActivationRecorder primitives per the task brief (does not go through a
    JacobianLens object -- there is no Jacobian here at all)."""
    input_ids = lm.encode(prompt)
    final_layer = lm.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    with ActivationRecorder(lm.layers, at=record_at) as recorder:
        lm.forward(input_ids)
        activations = {i: recorder.activations[i].detach() for i in record_at}

    def select(layer_idx: int) -> torch.Tensor:
        full = activations[layer_idx][0]  # [seq_len, d_model]
        return full[[position]].float()

    logit_lens_logits = {layer: lm.unembed(select(layer)).float().cpu() for layer in layers}
    model_logits = lm.unembed(select(final_layer)).float().cpu()
    return logit_lens_logits, model_logits, input_ids


def run_for_model(lm, tokenizer, label: str, prompts: list[dict]) -> list[dict]:
    results = []
    for p in prompts:
        formatted = format_prompt(tokenizer, p["text"])
        lens_logits, model_logits, input_ids = logit_lens_readout(
            lm, formatted, TARGET_LAYERS, position=-1
        )
        seq_len = input_ids.shape[1]
        entry = {
            "model_label": label,
            "prompt_type": p["type"],
            "prompt_text": p["text"],
            "formatted_prompt": formatted,
            "seq_len": seq_len,
            "position": seq_len - 1,
            "layers": {},
            "model_final_logits_top_k": [],
        }
        for layer in TARGET_LAYERS:
            logits = lens_logits[layer][0]
            topk = logits.topk(TOP_K)
            entry["layers"][str(layer)] = [
                {"rank": i + 1, "token": tokenizer.decode([tid]), "logit": float(val)}
                for i, (tid, val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()))
            ]
        topk_final = model_logits[0].topk(TOP_K)
        entry["model_final_logits_top_k"] = [
            {"rank": i + 1, "token": tokenizer.decode([tid]), "logit": float(val)}
            for i, (tid, val) in enumerate(
                zip(topk_final.indices.tolist(), topk_final.values.tolist())
            )
        ]
        results.append(entry)
        print(f"\n[{label}] prompt_type={p['type']!r} prompt={p['text']!r}")
        for layer in TARGET_LAYERS:
            toks = [t["token"] for t in entry["layers"][str(layer)]]
            print(f"  Plain logit-lens layer {layer} top-{TOP_K}: {toks}")
        print(f"  Model final-layer top-{TOP_K}: {[t['token'] for t in entry['model_final_logits_top_k']]}")
    return results


def main():
    token = load_hf_token()

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

    prompts = get_shared_prompts()

    all_results = []

    print("\n" + "#" * 70)
    print(f"# PLAIN LOGIT LENS -- MODEL UNDER TEST: {FEMALE_ADAPTER_ID}")
    print("#" * 70)
    lm_female = jlens.from_hf(female_model.get_base_model(), tokenizer)
    all_results += run_for_model(lm_female, tokenizer, FEMALE_ADAPTER_ID, prompts)

    if torch.cuda.is_available():
        print(f"\n[GPU] max_memory_allocated = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        print(f"[GPU] max_memory_reserved  = {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")

    print("\n" + "#" * 70)
    print(f"# PLAIN LOGIT LENS -- MODEL UNDER TEST: {BASE_MODEL_ID} (adapter disabled)")
    print("#" * 70)
    with female_model.disable_adapter():
        lm_base = jlens.from_hf(female_model.get_base_model(), tokenizer)
        all_results += run_for_model(
            lm_base, tokenizer, f"{BASE_MODEL_ID} (adapter disabled)", prompts
        )

    if torch.cuda.is_available():
        print(f"\n[GPU] max_memory_allocated = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        print(f"[GPU] max_memory_reserved  = {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")

    out_path = OUTPUTS_DIR / "logit_lens_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_layers": TARGET_LAYERS,
                "top_k": TOP_K,
                "method": "plain_logit_lens_no_jacobian_transport",
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
