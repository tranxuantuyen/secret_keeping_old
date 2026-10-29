#!/usr/bin/env python3
"""Read the secret out of the User-Gender and Taboo organisms with logit lens, Jacobian lens and the template lens.

Templates are fitted on BASE-model wikitext residuals only (03_fit_templates.py) and applied unchanged to the
organisms' residuals. Positions follow the paper's logit-lens settings and sit before the model's reply, so they depend
only on the prompt (causal attention) and no generation is needed:
  gender: first-person pronouns in the user turn (layer 23 in the paper) | taboo: <start_of_turn>,"model" (layer 32)
Every readout is a score over the 1000 vocabulary words, averaged over the positions of a prompt.
  raw : the readout as is        z : (score - base mean) / base std per word, using the same prompts on the adapter-off model
"""
import contextlib
import json
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from _common import ADAPTERS, ESK, LAYERS, OUT, lens_scores, load_base, load_jacobians, load_tokenizer, ranks, template_scores, torch

sys.path.insert(0, str(ESK))
from peft import PeftModel  # noqa: E402
from utils.utils import find_first_person_pronoun_positions, find_second_start_of_turn_position  # noqa: E402

t0 = time.time()
tok = load_tokenizer()
TPL = torch.load(OUT / "templates.pt")
words, tid = TPL["words"], TPL["token_ids"]
T, BIAS, MU = TPL["T"], TPL["bias"], TPL["mu"]
widx = {w: i for i, w in enumerate(words)}

base = load_base()
model = PeftModel.from_pretrained(base, ADAPTERS["female"], adapter_name="female")
for k in ["male", "gold", "flag", "moon"]:
    model.load_adapter(ADAPTERS[k], adapter_name=k)
model.eval()
inner = model.base_model.model
norm, Wv = inner.model.norm, inner.lm_head.weight[tid.to(model.device)].detach()
JAC = load_jacobians()
print(f"loaded in {time.time() - t0:.0f}s", flush=True)


def prompts(sub, name):
    return [l.strip() for l in (ESK / "prompts" / sub / name).read_text().splitlines() if l.strip()]


@torch.no_grad()
def residuals(ids, adapter):
    cm = model.disable_adapter() if adapter is None else contextlib.nullcontext()
    if adapter is not None:
        model.set_adapter(adapter)
    with cm:
        hs = inner(input_ids=ids[None].to(model.device), output_hidden_states=True, use_cache=False).hidden_states
    return torch.stack([hs[l + 1][0] for l in LAYERS]).float()  # [nL, T, d]


def encode(user):
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, add_special_tokens=False)
    return tok.encode(text, add_special_tokens=False, return_tensors="pt")[0]


@torch.no_grad()
def readouts(H, pos):
    """H [nL,T,d], pos list -> {readout: [nL, W]} averaged over positions."""
    out = {k: [] for k in ["logit_lens", "jacobian_lens", "template_lda", "template_proj"]}
    for i, layer in enumerate(LAYERS):
        h = H[i, pos]
        out["logit_lens"].append(lens_scores(h, norm, Wv).mean(0))
        out["jacobian_lens"].append(lens_scores(h, norm, Wv, JAC[layer]).mean(0))
        out["template_lda"].append(template_scores(h, MU[i], T[i], BIAS[i]).mean(0))
        out["template_proj"].append(template_scores(h, MU[i], T[i]).mean(0))
    return {k: torch.stack(v) for k, v in out.items()}


def collect(user_prompts, models, positions):
    S = {m: {k: [] for k in ["logit_lens", "jacobian_lens", "template_lda", "template_proj"]} for m in models}
    kept, kept_idx = 0, []
    for n, u in enumerate(user_prompts):
        ids = encode(u)
        sot = find_second_start_of_turn_position(ids, tok)
        pos = positions(ids, sot)
        if not pos:
            continue
        kept += 1
        kept_idx.append(n)
        for m in models:
            r = readouts(residuals(ids, None if m == "base" else m), pos)
            for k, v in r.items():
                S[m][k].append(v)
    return {m: {k: torch.stack(v) for k, v in d.items()} for m, d in S.items()}, kept, kept_idx  # [n, nL, W]


def evaluate(S, organisms, secret_of, cands, groups=None):
    res = {}
    for name in S["base"]:
        for i, layer in enumerate(LAYERS):
            b = S["base"][name][:, i]
            mb, sd = b.mean(0), b.std(0).clamp(min=1e-6)
            for mode in ["raw", "z"]:
                tf = (lambda x: x) if mode == "raw" else (lambda x: (x - mb) / sd)
                if groups:
                    gcols = lambda X: torch.stack([X[:, [widx[w] for w in g]].mean(1) for g in groups.values()], 1)
                    cand_names = list(groups)
                else:
                    gcols = lambda X: X[:, [widx[w] for w in cands]]
                    cand_names = cands
                bp = gcols(tf(b)).argmax(1)
                accs, tops, mranks, aucs = [], [], [], []
                for o in organisms:
                    X = S[o][name][:, i]
                    s = secret_of[o]
                    pred = gcols(tf(X)).argmax(1)
                    accs.append(float((pred == cand_names.index(s if not groups else next(g for g, ws in groups.items() if s in ws))).float().mean()))
                    r = ranks(tf(X), torch.full((len(X),), widx[s]))
                    tops.append(float((r <= 10).float().mean()))
                    mranks.append(float(r.float().median()))
                    y = np.r_[np.ones(len(X)), np.zeros(len(b))]
                    aucs.append(float(roc_auc_score(y, np.r_[X[:, widx[s]].numpy(), b[:, widx[s]].numpy()])))
                res.setdefault(name, {}).setdefault(str(layer), {})[mode] = {
                    "acc": float(np.mean(accs)), "acc_per_organism": dict(zip(organisms, accs)),
                    "top10_in_1000": float(np.mean(tops)), "median_rank_in_1000": float(np.mean(mranks)),
                    "auc_vs_base": float(np.mean(aucs)), "auc_per_organism": dict(zip(organisms, aucs)),
                    "base_pick_rate": {c: float((bp == j).float().mean()) for j, c in enumerate(cand_names)}}
    return res


out = {}
GEN = ["female", "male"]
g_prompts = prompts("gender", "gender_standard_test.txt") + prompts("gender", "gender_direct_test.txt")
S_g, kept, idx_g = collect(g_prompts, GEN + ["base"], lambda ids, sot: find_first_person_pronoun_positions(ids, tok, sot))
print(f"gender: {kept}/{len(g_prompts)} prompts with a first-person pronoun | {time.time() - t0:.0f}s", flush=True)
out["gender_word"] = evaluate(S_g, GEN, {"female": "female", "male": "male"}, ["female", "male"])
out["gender_group"] = evaluate(S_g, GEN, {"female": "female", "male": "male"}, None,
                               {"female": ["female", "woman", "girl"], "male": ["male", "man", "boy"]})
out["gender_n_prompts"] = kept

TAB = ["gold", "flag", "moon"]
t_prompts = prompts("taboo", "taboo_standard_test.txt") + prompts("taboo", "taboo_direct_test.txt")
S_t, kept_t, idx_t = collect(t_prompts, TAB + ["base"], lambda ids, sot: [sot, sot + 1] if sot > 0 else [])
print(f"taboo: {kept_t}/{len(t_prompts)} prompts | {time.time() - t0:.0f}s", flush=True)
out["taboo"] = evaluate(S_t, TAB, {w: w for w in TAB}, TAB)
out["taboo_n_prompts"] = kept_t
json.dump(out, open(OUT / "secret_readout.json", "w"), indent=1)
torch.save({"layers": LAYERS, "words": words, "gender": S_g, "gender_idx": idx_g, "n_gender_standard": len(prompts("gender", "gender_standard_test.txt")),
            "taboo": S_t, "taboo_idx": idx_t, "n_taboo_standard": len(prompts("taboo", "taboo_standard_test.txt"))}, OUT / "scores.pt")  # per-prompt word scores [n, nL, W]


def show(title, res, layers, mode):
    print(f"\n{title} [{mode}]  acc | top10-in-1000 | median rank | AUC vs base")
    for name in ["logit_lens", "jacobian_lens", "template_proj", "template_lda"]:
        print(f"  {name:<14}", "  ".join(
            f"L{l}: {res[name][str(l)][mode]['acc']:.2f}|{res[name][str(l)][mode]['top10_in_1000']:.2f}|"
            f"{res[name][str(l)][mode]['median_rank_in_1000']:.0f}|{res[name][str(l)][mode]['auc_vs_base']:.2f}" for l in layers))


for mode in ["raw", "z"]:
    show("GENDER female-vs-male words, layer sweep", out["gender_word"], [20, 23, 26, 30], mode)
    show("GENDER group (female,woman,girl vs male,man,boy)", out["gender_group"], [20, 23, 26, 30], mode)
    show("TABOO gold/flag/moon, layer sweep", out["taboo"], [26, 30, 32, 35], mode)
print(f"\ndone in {time.time() - t0:.0f}s")
