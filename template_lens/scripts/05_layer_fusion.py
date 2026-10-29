#!/usr/bin/env python3
"""Combine several layers' word readouts (rank fusion and relatives) and compare with the single paper layer.

Input : outputs/scores.pt (per-prompt scores over the 1000 words, every layer, every lens; written by 04_eval_secrets.py)
Output: outputs/layer_fusion.json

Fusion methods (applied to the layers of a fixed set; higher fused score = more likely the secret):
  rrf10 / rrf60 : sum over layers of 1 / (k + rank_layer(word))      reciprocal rank fusion
  borda         : minus the mean rank over layers
  persist50     : number of layers in which the word is in the top 50 (ties broken by rrf60)
  zavg          : mean over layers of the per-layer z-score (z mode only)
Modes: raw = scores as they are; z = every (layer, word) score is first standardised with the base model's mean/std on the same prompts.

Everything below is fixed before looking at results: the layer sets, k, the methods. The only selection is done by
choosing the best config on the STANDARD prompts and reporting it on the DIRECT prompts.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from _common import LAYERS, OUT, torch

D = torch.load(OUT / "scores.pt")
words = D["words"]
widx = {w: i for i, w in enumerate(words)}
LI = {l: i for i, l in enumerate(LAYERS)}
READ = ["logit_lens", "jacobian_lens", "template_proj", "template_lda"]
GROUPS = {"female": ["female", "woman", "girl"], "male": ["male", "man", "boy"]}
GSETS = {"around 20-23-26": [20, 23, 26], "before 15-20-23": [15, 20, 23], "spread 15-23-30": [15, 23, 30], "all 8 layers": LAYERS}
TASKS = {
    "gender_word": dict(src="gender", orgs=["female", "male"], cands=["female", "male"], groups=None, target=23, sets=GSETS),
    "gender_group": dict(src="gender", orgs=["female", "male"], cands=None, groups=GROUPS, target=23, sets=GSETS),
    "taboo": dict(src="taboo", orgs=["gold", "flag", "moon"], cands=["gold", "flag", "moon"], groups=None, target=32,
                  sets={"before 26-30-32": [26, 30, 32], "around 30-32-35": [30, 32, 35], "spread 20-26-32": [20, 26, 32], "all 8 layers": LAYERS}),
}
METHODS = {"raw": ["rrf10", "rrf60", "borda", "persist50"], "z": ["zavg", "rrf60", "borda", "persist50"]}
BOOT = 2000


def to_ranks(X):
    """1-based rank of every word within its (prompt, layer) row; rank 1 = highest score."""
    order = X.argsort(-1, descending=True)
    pos = torch.arange(1, X.shape[-1] + 1, dtype=torch.float32).expand_as(order).contiguous()
    return torch.empty_like(X).scatter_(-1, order, pos)


def prep(S, readout, mode, models):
    b = S["base"][readout]
    if mode == "z":
        mb, sd = b.mean(0), b.std(0).clamp(min=1e-6)
        return {m: (S[m][readout] - mb) / sd for m in models}
    return {m: S[m][readout] for m in models}


def fuse(X, R, ids, method):
    """X, R: [n, nL, W] scores and ranks; ids: layer indices -> fused score [n, W]."""
    if method == "single":
        return X[:, ids[0]]
    Rs = R[:, ids]
    rrf = lambda k: (1.0 / (k + Rs)).sum(1)
    if method == "rrf10":
        return rrf(10)
    if method == "rrf60":
        return rrf(60)
    if method == "borda":
        return -Rs.mean(1)
    if method == "persist50":
        return (Rs <= 50).sum(1).float() + 0.01 * rrf(60)
    if method == "zavg":
        return X[:, ids].mean(1)
    raise ValueError(method)


def evaluate(F, Fb, T, mask):
    if T["groups"]:
        names = list(T["groups"])
        cols = [[widx[w] for w in g] for g in T["groups"].values()]
        gc = lambda X: torch.stack([X[:, c].mean(1) for c in cols], 1)
    else:
        names = T["cands"]
        cols = [widx[w] for w in names]
        gc = lambda X: X[:, cols]
    Fb = Fb[mask]
    bp = gc(Fb).argmax(1)
    acc, top10, med, auc, corr = {}, [], [], [], {}
    for o in T["orgs"]:
        X = F[o][mask]
        c = gc(X).argmax(1) == names.index(o)
        corr[o] = c.numpy()
        acc[o] = float(c.float().mean())
        r = 1 + (X > X[:, widx[o]:widx[o] + 1]).sum(1)
        top10.append(float((r <= 10).float().mean()))
        med.append(float(r.float().median()))
        y = np.r_[np.ones(len(X)), np.zeros(len(Fb))]
        auc.append(float(roc_auc_score(y, np.r_[X[:, widx[o]].numpy(), Fb[:, widx[o]].numpy()])))
    m = {"acc": float(np.mean(list(acc.values()))), "acc_per_organism": acc, "top10_in_1000": float(np.mean(top10)),
         "median_rank_in_1000": float(np.mean(med)), "auc_vs_base": float(np.mean(auc)),
         "base_pick_rate": {c: float((bp == j).float().mean()) for j, c in enumerate(names)}}
    return m, corr


rng = np.random.default_rng(0)
res = {}
for tname, T in TASKS.items():
    S = D[T["src"]]
    n = S["base"][READ[0]].shape[0]
    is_std = torch.tensor(np.array(D[T["src"] + "_idx"]) < D["n_" + T["src"] + "_standard"])
    masks = {"all": torch.ones(n, dtype=torch.bool), "standard": is_std, "direct": ~is_std}
    boot_idx = rng.integers(0, n, (BOOT, n))
    tl = f"single|L{T['target']}"
    for mode in ["raw", "z"]:
        for rd in READ:
            P = prep(S, rd, mode, T["orgs"] + ["base"])
            R = {m: to_ranks(P[m]) for m in P}
            configs = {f"single|L{l}": ("single", [l]) for l in LAYERS}
            for sname, ls in T["sets"].items():
                for meth in METHODS[mode]:
                    configs[f"{meth}|{sname}"] = (meth, ls)
            out, corr_all = {}, {}
            for cname, (meth, ls) in configs.items():
                ids = [LI[l] for l in ls]
                F = {o: fuse(P[o], R[o], ids, meth) for o in T["orgs"]}
                Fb = fuse(P["base"], R["base"], ids, meth)
                out[cname] = {}
                for sub, mask in masks.items():
                    out[cname][sub], c = evaluate(F, Fb, T, mask)
                    if sub == "all":
                        corr_all[cname] = np.stack([c[o] for o in T["orgs"]]).astype(float)  # [orgs, n]
            base_c = corr_all[tl]
            for cname in configs:
                d = corr_all[cname] - base_c
                bd = d[:, boot_idx].mean(2).mean(0)  # [BOOT]
                out[cname]["all"]["acc_diff_vs_paper_layer"] = {
                    "diff": float(d.mean()), "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5))}
            pool = [c for c in configs if not c.startswith("single")]
            best = max(pool, key=lambda c: out[c]["standard"]["acc"])
            best_single = max((c for c in configs if c.startswith("single")), key=lambda c: out[c]["standard"]["acc"])
            out["_selected"] = {
                "fused_config_chosen_on_standard": best,
                "fused_standard_acc": out[best]["standard"]["acc"], "fused_direct_acc": out[best]["direct"]["acc"],
                "paper_layer_standard_acc": out[tl]["standard"]["acc"], "paper_layer_direct_acc": out[tl]["direct"]["acc"],
                "best_single_layer_chosen_on_standard": best_single,
                "best_single_standard_acc": out[best_single]["standard"]["acc"], "best_single_direct_acc": out[best_single]["direct"]["acc"]}
            res.setdefault(tname, {}).setdefault(mode, {})[rd] = out
    print(f"{tname}: done", flush=True)

json.dump(res, open(OUT / "layer_fusion.json", "w"), indent=1)


def grid(tname, mode):
    T = TASKS[tname]
    tl = f"single|L{T['target']}"
    sets = list(T["sets"])
    print(f"\n=== {tname} [{mode}]  accuracy over all prompts; first column = paper layer L{T['target']}; * = 95% CI of the gain over it excludes 0")
    print(f"{'readout':<14}{'method':<10}{'L' + str(T['target']):>7}  " + "  ".join(f"{s:>17}" for s in sets))
    for rd in READ:
        o = res[tname][mode][rd]
        for k, meth in enumerate(METHODS[mode]):
            cells = []
            for s in sets:
                e = o[f"{meth}|{s}"]["all"]
                ci = e["acc_diff_vs_paper_layer"]
                cells.append(f"{e['acc']:.2f}{'*' if ci['lo'] > 0 else ('-' if ci['hi'] < 0 else ' ')}")
            print(f"{rd if k == 0 else '':<14}{meth:<10}{o[tl]['all']['acc']:>7.2f}  " + "  ".join(f"{c:>17}" for c in cells))


def selected(tname, mode):
    T = TASKS[tname]
    print(f"\n--- {tname} [{mode}]  choose on STANDARD prompts, report on DIRECT prompts")
    print(f"{'readout':<14}{'chosen fused config':<34}{'std':>6}{'direct':>8}   {'paper layer direct':>20}   {'best single layer (chosen on std)':>34}")
    for rd in READ:
        s = res[tname][mode][rd]["_selected"]
        print(f"{rd:<14}{s['fused_config_chosen_on_standard']:<34}{s['fused_standard_acc']:>6.2f}{s['fused_direct_acc']:>8.2f}   "
              f"{s['paper_layer_direct_acc']:>20.2f}   {s['best_single_layer_chosen_on_standard'] + ' -> ' + format(s['best_single_direct_acc'], '.2f'):>34}")


for tname in TASKS:
    for mode in ["raw", "z"]:
        grid(tname, mode)
for tname in TASKS:
    for mode in ["raw", "z"]:
        selected(tname, mode)
