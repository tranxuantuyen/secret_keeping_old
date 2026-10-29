"""
Shared environment / path setup for all demo scripts in this project.

Import this module FIRST, before importing `transformers`, `datasets`, or
`huggingface_hub`, so that the HF cache environment variables are set before
those libraries resolve their default cache locations.

All paths are derived from the project root via pathlib, so the project
stays portable (no hardcoded absolute strings baked into demo logic).
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root is the parent of this file's directory (demos/ -> secret_keeping/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_REPO = PROJECT_ROOT / "external" / "eliciting-secret-knowledge"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
HF_HOME_DIR = CACHE_DIR / "huggingface"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
HF_DATASETS_CACHE_DIR = HF_HOME_DIR / "datasets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- HF cache env vars -----------------------------------------------------
# NOTE on avoiding double-caching:
# In current huggingface_hub (0.36) / transformers (4.57), HF_HUB_CACHE
# defaults to "<HF_HOME>/hub" when not set explicitly, and transformers'
# model cache resolves through huggingface_hub's cache (TRANSFORMERS_CACHE
# is a deprecated legacy alias that, if left unset, also falls back to the
# same HF_HOME-derived location). The failure mode from the prior attempt
# was setting TRANSFORMERS_CACHE to a *different* directory than HF_HOME/hub,
# which causes transformers and huggingface_hub to each maintain their own
# ~18GB copy of the base model. We avoid that here by:
#   1. Setting HF_HOME explicitly.
#   2. Setting HF_HUB_CACHE explicitly to "<HF_HOME>/hub" (the same path
#      HF_HOME would derive by default) so every library that reads either
#      variable agrees on one physical location.
#   3. NOT setting TRANSFORMERS_CACHE as a separate location. If a script
#      needs it for backward compatibility with older code, it should be
#      set equal to HF_HUB_CACHE, never to a different path.
os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))
# Kept equal to HF_HUB_CACHE on purpose -- see note above.
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
