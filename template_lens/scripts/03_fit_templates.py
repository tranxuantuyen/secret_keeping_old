#!/usr/bin/env python3
"""Fit template vectors t_w = (Sigma + lam I)^-1 (mu_w - mu) per layer, pick lam on held-out wikitext, and compare readouts.

Task on held-out contexts: rank the true next word among the 1000 vocabulary words using
  logit lens, Jacobian lens (J_l from neuronpedia), template projection t_w.(h-mu), template LDA t_w.(h-(mu+mu_w)/2).
"""
import json
import time

from _common import LAYERS, OUT, load_base, load_jacobians, lens_scores, ranks, solve_templates, template_scores, torch

LAMS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
t0 = time.time()
S = torch.load(OUT / "stats.pt")
words, tid = S["words"], S["token_ids"]
held, hw = S["held"], S["held_word"].long()
model = load_base()
norm = model.model.norm
Wv = model.lm_head.weight[tid.to(model.device)].detach()
JAC = load_jacobians()
print(f"loaded in {time.time() - t0:.0f}s | held-out contexts {len(hw)}", flush=True)


def metrics(sc):
    r = ranks(sc, hw)
    return {"top1": float((r == 1).float().mean()), "top10": float((r <= 10).float().mean()),
            "mrr": float((1.0 / r.float()).mean()), "median_rank": float(r.float().median())}


res = {"lens": {}, "lam_sweep": {}, "chosen_lam": {}, "per_word_top1": {}}
T_all, B_all, MU_all = [], [], []
for i, layer in enumerate(LAYERS):
    h = held[:, i]
    res["lens"].setdefault("logit_lens", {})[layer] = metrics(lens_scores(h, norm, Wv))
    res["lens"].setdefault("jacobian_lens", {})[layer] = metrics(lens_scores(h, norm, Wv, JAC[layer]))
    mu, cov, wmean = S["mu"][i], S["cov"][i], S["wmean"][i]
    best = None
    for lam in LAMS:
        T, bias = solve_templates(mu, cov, wmean, lam)
        m_lda = metrics(template_scores(h, mu, T, bias))
        res["lam_sweep"].setdefault(str(layer), {})[str(lam)] = {"lda": m_lda, "proj": metrics(template_scores(h, mu, T))}
        if best is None or m_lda["mrr"] > best[0]:
            best = (m_lda["mrr"], lam, T, bias)
    _, lam, T, bias = best
    res["chosen_lam"][layer] = lam
    res["lens"].setdefault("template_lda", {})[layer] = res["lam_sweep"][str(layer)][str(lam)]["lda"]
    res["lens"].setdefault("template_proj", {})[layer] = res["lam_sweep"][str(layer)][str(lam)]["proj"]
    T_all.append(T.cpu()); B_all.append(bias.cpu()); MU_all.append(mu)
    print(f"L{layer:>2} lam*={lam:g} | " + " | ".join(f"{k} top1={v[layer]['top1']:.3f} mrr={v[layer]['mrr']:.3f}"
                                                   for k, v in res["lens"].items()), flush=True)

# per-word top-1 for the repo's secret words at the best layer for each readout
focus = ["female", "male", "woman", "man", "girl", "boy", "gold", "flag", "moon"]
for i, layer in enumerate(LAYERS):
    h = held[:, i]
    T, bias, mu = T_all[i], B_all[i], MU_all[i]
    scs = {"logit_lens": lens_scores(h, norm, Wv), "jacobian_lens": lens_scores(h, norm, Wv, JAC[layer]),
           "template_lda": template_scores(h, mu, T.cuda(), bias.cuda())}
    for name, sc in scs.items():
        r = ranks(sc, hw)
        for w in focus:
            m = hw == words.index(w)
            res["per_word_top1"].setdefault(name, {}).setdefault(str(layer), {})[w] = float((r[m] == 1).float().mean())

torch.save({"layers": LAYERS, "words": words, "token_ids": tid, "T": torch.stack(T_all), "bias": torch.stack(B_all),
            "mu": torch.stack(MU_all), "lam": res["chosen_lam"]}, OUT / "templates.pt")
json.dump(res, open(OUT / "wikitext_heldout.json", "w"), indent=1)
print(f"saved templates.pt + wikitext_heldout.json in {time.time() - t0:.0f}s")
