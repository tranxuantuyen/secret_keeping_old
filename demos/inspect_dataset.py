#!/usr/bin/env python3
"""
Inspect the released User Gender datasets (read-only).

Datasets (all `messages`-format HF datasets of {role, content} turns):
  - bcywinski/user-gender-model        : how the model should answer when
                                          directly asked about ITS OWN gender
                                          (not the user's).
  - bcywinski/user-gender-female       : training examples with gender-coded
                                          answers consistent with "user is female"
  - bcywinski/user-gender-male         : same, consistent with "user is male"
  - bcywinski/user-gender-adversarial  : training examples where the model is
                                          asked (adversarially) about the USER's
                                          gender and should deflect/deny knowledge
  - bcywinski/female-validate          : A/B multiple-choice eval set, gold
                                          answer consistent with female user
  - bcywinski/male-validate            : same, gold answer consistent with male

Usage:
    python demos/inspect_dataset.py
    python demos/inspect_dataset.py --dataset bcywinski/female-validate --n 5
"""

import argparse

import _env_setup  # noqa: F401
from datasets import load_dataset

from _env_setup import load_hf_token

ALL_DATASETS = [
    "bcywinski/user-gender-model",
    "bcywinski/user-gender-female",
    "bcywinski/user-gender-male",
    "bcywinski/user-gender-adversarial",
    "bcywinski/female-validate",
    "bcywinski/male-validate",
]


def print_example(name: str, idx: int, example: dict):
    messages = example["messages"]
    print("-" * 70)
    print(f"[{name}] example #{idx}")
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"{role}:\n  {content}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        help="Specific dataset repo id to inspect. If omitted, inspects all known gender datasets.",
    )
    parser.add_argument("--n", type=int, default=3, help="Number of examples to print per dataset.")
    args = parser.parse_args()

    token = load_hf_token()

    datasets_to_show = [args.dataset] if args.dataset else ALL_DATASETS

    for name in datasets_to_show:
        print("=" * 70)
        print(f"DATASET: {name}")
        print("=" * 70)
        ds = load_dataset(name, split="train", token=token)
        print(f"Num rows: {len(ds)}")
        print(f"Columns:  {ds.column_names}")
        for i in range(min(args.n, len(ds))):
            print_example(name, i, ds[i])
        print()


if __name__ == "__main__":
    main()
