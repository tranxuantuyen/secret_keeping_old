#!/usr/bin/env python3
"""
Inspect the User Gender organism model(s) without doing any training,
merging, or steering. Read-only inspection:

- base model identity (read from adapter_config.json)
- PEFT/LoRA adapter metadata (rank, alpha, target modules, trainable params)
- tokenizer class / vocab size
- transformer architecture (num layers, hidden size)

Usage:
    python demos/inspect_model.py --model female
    python demos/inspect_model.py --model male
    python demos/inspect_model.py --model base
"""

import argparse
import json

import _env_setup  # noqa: F401  (must be imported first to set cache env vars)
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from _env_setup import BASE_MODEL_ID, FEMALE_ADAPTER_ID, MALE_ADAPTER_ID, load_hf_token


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def inspect_adapter_config(adapter_repo: str, token: str):
    path = hf_hub_download(adapter_repo, "adapter_config.json", token=token)
    with open(path) as f:
        cfg = json.load(f)
    print(f"\n--- adapter_config.json for {adapter_repo} ---")
    print(json.dumps(cfg, indent=2))
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["female", "male", "base"],
        default="female",
        help="Which model to load: the user-female adapter, user-male adapter, or plain base model.",
    )
    args = parser.parse_args()

    token = load_hf_token()

    adapter_repo = {
        "female": FEMALE_ADAPTER_ID,
        "male": MALE_ADAPTER_ID,
        "base": None,
    }[args.model]

    print("=" * 70)
    print(f"Inspecting: {args.model} ({adapter_repo or BASE_MODEL_ID})")
    print("=" * 70)

    if adapter_repo is not None:
        adapter_cfg = inspect_adapter_config(adapter_repo, token)
        base_model_id = adapter_cfg["base_model_name_or_path"]
        assert base_model_id == BASE_MODEL_ID, (
            f"Unexpected base model in adapter_config.json: {base_model_id}"
        )
    else:
        base_model_id = BASE_MODEL_ID

    print(f"\nLoading tokenizer for base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)

    print(f"Loading base model: {base_model_id} (bf16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
    )

    # Must count BEFORE wrapping: PeftModel.from_pretrained injects LoRA
    # layers into `base_model` in place, so counting after wrapping would
    # make the "base" count already include the adapter's parameters.
    base_total, _ = count_params(base_model)

    if adapter_repo is not None:
        print(f"Attaching PEFT/LoRA adapter (unmerged): {adapter_repo}")
        model = PeftModel.from_pretrained(base_model, adapter_repo, token=token)
    else:
        model = base_model

    model.eval()

    total_params, trainable_params = count_params(model)

    config = base_model.config
    print("\n--- Summary ---")
    print(f"Model class:            {type(model).__name__}")
    print(f"Base model:              {base_model_id}")
    print(f"Dtype:                   {next(model.parameters()).dtype}")
    print(f"Total parameters:        {total_params:,}")
    print(f"Base model parameters:   {base_total:,}")
    if adapter_repo is not None:
        # LoRA adds a small number of extra parameters on top of the frozen base.
        adapter_params = total_params - base_total
        print(f"Adapter (LoRA) params:   {adapter_params:,} (delta vs. base)")
    print(f"Tokenizer class:         {type(tokenizer).__name__}")
    print(f"Vocab size:              {tokenizer.vocab_size:,}")
    print(f"Num transformer layers:  {config.num_hidden_layers}")
    print(f"Hidden size:             {config.hidden_size}")
    print(f"Num attention heads:     {config.num_attention_heads}")
    print(f"Num KV heads:            {getattr(config, 'num_key_value_heads', 'n/a')}")

    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"\nGPU max memory allocated: {allocated:.2f} GiB")
        print(f"GPU max memory reserved:  {reserved:.2f} GiB")


if __name__ == "__main__":
    main()
