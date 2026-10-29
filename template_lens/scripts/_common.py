"""Shared paths, environment setup and helpers for the template-lens scripts (run with the eval_aware python)."""
from __future__ import annotations

import os
import site
import sys
import warnings
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
ESK = ROOT / "external" / "eliciting-secret-knowledge"
OUT = PKG / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

HF = ROOT / "cache" / "huggingface"
for _k, _v in {"HF_HOME": HF, "HF_HUB_CACHE": HF / "hub", "HF_DATASETS_CACHE": HF / "datasets",
               "TRANSFORMERS_CACHE": HF / "hub"}.items():
    os.environ.setdefault(_k, str(_v))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

# ~/.local holds a broken httpx; env packages must win, user site only fills gaps (torch, datasets).
_us = site.getusersitepackages()
sys.path = [p for p in sys.path if p != _us] + [_us]

import torch  # noqa: E402

BASE_ID = "google/gemma-2-9b-it"
ADAPTERS = {
    "female": "bcywinski/gemma-2-9b-it-user-female",
    "male": "bcywinski/gemma-2-9b-it-user-male",
    "gold": "bcywinski/gemma-2-9b-it-taboo-gold",
    "flag": "bcywinski/gemma-2-9b-it-taboo-flag",
    "moon": "bcywinski/gemma-2-9b-it-taboo-moon",
}
JLENS_REPO = "neuronpedia/jacobian-lens"
JLENS_FILE = "gemma-2-9b-it/jlens/Salesforce-wikitext/gemma-2-9b-it_jacobian_lens.pt"
LAYERS = [10, 15, 20, 23, 26, 30, 32, 35]  # 23 = gender lens layer, 32 = Taboo lens layer
LAMBDA_REL = 1e-2  # ridge strength relative to mean eigenvalue of the covariance


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_ID)


def load_base():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16, device_map="auto").eval()


def load_jacobians(device="cuda"):
    from huggingface_hub import hf_hub_download
    ck = torch.load(hf_hub_download(JLENS_REPO, JLENS_FILE), map_location="cpu", weights_only=True)
    return {l: ck["J"][l].to(device=device, dtype=torch.float32) for l in LAYERS}


def solve_templates(mu, cov, wmean, lam_rel=LAMBDA_REL):
    """t_w = (Sigma + lam I)^-1 (mu_w - mu) for every word at one layer. Returns (T [W,d], lda_bias [W])."""
    cov = cov.double().cuda()
    mu = mu.double().cuda()
    d = cov.shape[0]
    lam = lam_rel * torch.trace(cov) / d
    A = cov + lam * torch.eye(d, dtype=torch.float64, device=cov.device)
    delta = wmean.double().cuda() - mu  # [W, d]
    T = torch.linalg.solve(A, delta.T).T
    return T.float(), (0.5 * (T * delta).sum(-1)).float()


@torch.no_grad()
def template_scores(h, mu, T, bias=None, chunk=4096):
    """Project residuals onto templates: t_w . (h - mu), optionally minus the LDA offset. h: [N, d]."""
    mu, T = mu.cuda().float(), T.cuda().float()
    bias = None if bias is None else bias.cuda().float()
    out = []
    for s in range(0, len(h), chunk):
        sc = (h[s:s + chunk].to("cuda").float() - mu) @ T.T
        out.append((sc if bias is None else sc - bias).cpu())
    return torch.cat(out)


@torch.no_grad()
def lens_scores(h, norm, Wv, J=None, chunk=4096):
    """Logit lens (J=None) or Jacobian lens (J = layer matrix) logits restricted to the vocabulary rows Wv [W, d]."""
    out = []
    for s in range(0, len(h), chunk):
        x = h[s:s + chunk].to("cuda").float()
        if J is not None:
            x = x @ J.T
        z = norm(x.to(torch.bfloat16))
        out.append((z.float() @ Wv.float().T).cpu())
    return torch.cat(out)


def ranks(scores, true_idx):
    """1-based rank of the true column among all columns (ties count in favour of the true word)."""
    t = scores.gather(1, true_idx.view(-1, 1))
    return 1 + (scores > t).sum(1)
