#!/usr/bin/env python3
"""
Apply the pre-fitted Anthropic Jacobian lens (jlens, model `google/gemma-2-9b-it`,
fit on `Salesforce-wikitext`, downloaded from `neuronpedia/jacobian-lens`) to
residual-stream activations of the User-Gender "female" secret-keeping
organism, and (for comparison) to the plain base model.

Read-only interpretability only: no steering, no activation editing, no
retraining, no new lens fit. Uses the SAME prompts, chat-template
construction, and final-prompt-token position convention as
demos/basic_gender_demo.py / demos/inspect_activations.py.

PEFT handling: jlens.from_hf() auto-detects the residual stack by walking
known attribute-path "layouts" (model.layers / model.norm / model.embed_tokens
etc.) starting from a bare `*ForCausalLM`. A `peft.PeftModel` wraps that in
`PeftModelForCausalLM`, which jlens's layout auto-detection does NOT recognize
(confirmed empirically: `jlens.from_hf(peft_model, tok)` raises ValueError).
`peft_model.get_base_model()` (equivalently `peft_model.base_model.model`,
verified to be the *same object*) returns the underlying `Gemma2ForCausalLM`
with LoRA adapters merged into the module tree *in place* (the adapter
modifies nn.Linear forward behavior without changing the module tree shape,
exactly as anticipated) -- jlens.from_hf() on that object succeeds and the
Jacobian lens correctly reads out POST-ADAPTER activations (LoRA deltas are
applied during the forward pass through those same Linear modules; only the
auto-detection step needed the unwrapped class name).

Must be run in the `jlens_env` conda environment (needs transformers>=5.5
and the jlens package; NOT compatible with eval_aware's transformers==4.57.6).

Usage:
    conda run -n jlens_env python scripts/apply_jlens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import _env_setup  # noqa: F401  (sets HF cache env vars first)
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

from _env_setup import (
    BASE_MODEL_ID,
    FEMALE_ADAPTER_ID,
    JLENS_FILENAME,
    JLENS_REPO_ID,
    HF_HUB_CACHE_DIR,
    OUTPUTS_DIR,
    get_shared_prompts,
    load_hf_token,
)

TARGET_LAYERS = [10, 23, 35]  # early / mid / late of 42 layers; 23 matches the
# cywinski repo's own TARGET_LAYER for this organism (user_gender/scripts/run_get_logit_lens.sh)
TOP_K = 10


def format_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )


def run_for_model(lens, lm, tokenizer, label: str, prompts: list[dict]) -> list[dict]:
    results = []
    for p in prompts:
        formatted = format_prompt(tokenizer, p["text"])
        lens_logits, model_logits, input_ids = lens.apply(
            lm, formatted, layers=TARGET_LAYERS, positions=[-1]
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
            print(f"  Jacobian-lens layer {layer} top-{TOP_K}: {toks}")
        print(f"  Model final-layer top-{TOP_K}: {[t['token'] for t in entry['model_final_logits_top_k']]}")
    return results


def main():
    token = load_hf_token()

    print("Downloading / locating pre-fitted Jacobian lens "
          f"({JLENS_REPO_ID}/{JLENS_FILENAME}) into local cache "
          f"{HF_HUB_CACHE_DIR} ...")
    lens_path = hf_hub_download(
        repo_id=JLENS_REPO_ID,
        filename=JLENS_FILENAME,
        cache_dir=str(HF_HUB_CACHE_DIR),
    )
    print(f"Lens file: {lens_path}")
    lens = jlens.JacobianLens.load(lens_path)
    print(f"Loaded lens: {lens}")
    missing = [l for l in TARGET_LAYERS if l not in lens.source_layers]
    if missing:
        raise RuntimeError(f"Target layers {missing} not in lens.source_layers={lens.source_layers}")

    print(f"\nLoading tokenizer: {BASE_MODEL_ID}")
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
    print(f"\nShared prompt set ({len(prompts)} prompts):")
    for p in prompts:
        print(f"  [{p['type']}] {p['text']!r}")

    all_results = []

    # --- User-female organism (PEFT-wrapped) ---
    print("\n" + "#" * 70)
    print(f"# JACOBIAN LENS -- MODEL UNDER TEST: {FEMALE_ADAPTER_ID}")
    print("#" * 70)
    lm_female = jlens.from_hf(female_model.get_base_model(), tokenizer)
    print(f"jlens.from_hf succeeded via peft_model.get_base_model(): {lm_female}")
    all_results += run_for_model(lens, lm_female, tokenizer, FEMALE_ADAPTER_ID, prompts)

    if torch.cuda.is_available():
        print(f"\n[GPU] max_memory_allocated = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        print(f"[GPU] max_memory_reserved  = {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")

    # --- Plain base model (adapter disabled, same weights, no reload) ---
    print("\n" + "#" * 70)
    print(f"# JACOBIAN LENS -- MODEL UNDER TEST: {BASE_MODEL_ID} (adapter disabled)")
    print("#" * 70)
    with female_model.disable_adapter():
        lm_base = jlens.from_hf(female_model.get_base_model(), tokenizer)
        all_results += run_for_model(
            lens, lm_base, tokenizer, f"{BASE_MODEL_ID} (adapter disabled)", prompts
        )

    if torch.cuda.is_available():
        print(f"\n[GPU] max_memory_allocated = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        print(f"[GPU] max_memory_reserved  = {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")

    out_path = OUTPUTS_DIR / "jlens_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "lens_repo_id": JLENS_REPO_ID,
                "lens_filename": JLENS_FILENAME,
                "lens_repr": repr(lens),
                "target_layers": TARGET_LAYERS,
                "top_k": TOP_K,
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
