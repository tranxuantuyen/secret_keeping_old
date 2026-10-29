"""
Shared environment / path setup for logit_lens_reproduction scripts.

Mirrors secret_keeping/demos/_env_setup.py (and
secret_keeping/jacobian_lens_comparison/scripts/_env_setup.py) in spirit:
sets HF cache env vars to the SAME physical cache used by the rest of
secret_keeping/ (secret_keeping/cache/huggingface/), before transformers/
datasets/huggingface_hub are imported, so nothing gets downloaded to
~/.cache, and provides a non-printing load_hf_token().

NOTE: this file is self-contained (does not try to `from _env_setup import
...` demos/_env_setup.py) because this module is ALSO named `_env_setup`,
which previously caused a circular self-import when sys.path had both
directories inserted (Python resolved `from _env_setup import ...` back to
this partially-initialized module itself, not to demos/_env_setup.py).
Being self-contained avoids that footgun entirely, matching how
jacobian_lens_comparison/scripts/_env_setup.py already does it.

Import this module FIRST, before importing `transformers`, `datasets`, or
`huggingface_hub`.
"""

from __future__ import annotations

import os
from pathlib import Path

# logit_lens_reproduction/scripts -> logit_lens_reproduction -> secret_keeping
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_REPO = PROJECT_ROOT / "external" / "eliciting-secret-knowledge"
CACHE_DIR = PROJECT_ROOT / "cache"
HF_HOME_DIR = CACHE_DIR / "huggingface"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
HF_DATASETS_CACHE_DIR = HF_HOME_DIR / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = COMPARISON_ROOT / "outputs"

# Same convention as demos/_env_setup.py: HF_HOME + HF_HUB_CACHE explicit,
# TRANSFORMERS_CACHE kept EQUAL to HF_HUB_CACHE (never a separate directory),
# to avoid the double-caching bug documented elsewhere in this project.
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE_DIR))

for _d in (MODELS_DIR, DATA_DIR, HF_HOME_DIR, HF_HUB_CACHE_DIR, HF_DATASETS_CACHE_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_hf_token() -> str:
    """Read HF_TOKEN from the project's .env file without ever printing it.

    Returns the token string. Raises RuntimeError if not found.
    """
    env_path = PROJECT_ROOT.parent / "code" / ".env"
    if not env_path.exists():
        raise RuntimeError(f"Expected .env file not found at {env_path}")

    token = None
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
                break

    if not token:
        raise RuntimeError("HF_TOKEN not found in .env file")

    # Make it available to libraries that read the env var directly,
    # without ever printing/logging the value itself.
    os.environ.setdefault("HF_TOKEN", token)
    return token


BASE_MODEL_ID = "google/gemma-2-9b-it"
FEMALE_ADAPTER_ID = "bcywinski/gemma-2-9b-it-user-female"
MALE_ADAPTER_ID = "bcywinski/gemma-2-9b-it-user-male"

AUDITOR_MODEL_ID = "google/gemma-3-4b-it"
TARGET_GENDER = "female"
