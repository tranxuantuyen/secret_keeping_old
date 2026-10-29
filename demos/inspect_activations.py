#!/usr/bin/env python3
"""
Plumbing sanity check ONLY: capture the residual-stream hidden state at the
final prompt token, for every layer, for a single prompt run through the
user-female model. This verifies the mechanics for reading hidden states --
it performs NO steering, probing, PCA, or Jacobian computation itself.

Saves a dict to outputs/activations_demo.pt:
    {
        "prompt": <formatted prompt text>,
        "model_id": "bcywinski/gemma-2-9b-it-user-female",
        "num_layers": int,
        "hidden_size": int,
        "hidden_states": torch.Tensor of shape [num_layers + 1, hidden_size]
            (layer 0 = embedding output, layers 1..N = after each transformer
            block; all taken at the final token position of the prompt, bf16,
            moved to CPU before saving),
    }

Usage:
    python demos/inspect_activations.py
"""

import _env_setup  # noqa: F401
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from _env_setup import BASE_MODEL_ID, FEMALE_ADAPTER_ID, OUTPUTS_DIR, load_hf_token


def main():
    token = load_hf_token()

    print(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=token)

    print(f"Loading base model: {BASE_MODEL_ID} (bf16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )

    print(f"Attaching PEFT/LoRA adapter (unmerged): {FEMALE_ADAPTER_ID}")
    model = PeftModel.from_pretrained(base_model, FEMALE_ADAPTER_ID, token=token)
    model.eval()

    prompt = "Do you know my gender?"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )
    print(f"\nFormatted prompt:\n{formatted_prompt!r}")

    inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # hidden_states is a tuple of length num_layers+1, each [batch, seq, hidden]
    hidden_states = outputs.hidden_states
    num_layers_plus_embed = len(hidden_states)
    hidden_size = hidden_states[0].shape[-1]

    print(f"\nNumber of hidden_states entries (embeddings + each layer): {num_layers_plus_embed}")
    print(f"Each entry shape: {tuple(hidden_states[0].shape)}  (batch, seq_len, hidden_size)")
    print(f"Hidden size: {hidden_size}")

    # Final prompt token position
    final_token_states = torch.stack(
        [h[0, -1, :].to(torch.float32).cpu() for h in hidden_states], dim=0
    )  # [num_layers+1, hidden_size]
    print(f"\nFinal-prompt-token hidden state stack shape: {tuple(final_token_states.shape)}")

    save_path = OUTPUTS_DIR / "activations_demo.pt"
    torch.save(
        {
            "prompt": formatted_prompt,
            "model_id": FEMALE_ADAPTER_ID,
            "num_layers": num_layers_plus_embed - 1,
            "hidden_size": hidden_size,
            "hidden_states": final_token_states,
        },
        save_path,
    )
    print(f"\nSaved activations to: {save_path}")

    if torch.cuda.is_available():
        print(f"\n[GPU] max_memory_allocated = {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        print(f"[GPU] max_memory_reserved  = {torch.cuda.max_memory_reserved() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
