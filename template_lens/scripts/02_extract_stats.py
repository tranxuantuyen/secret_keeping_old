#!/usr/bin/env python3
"""Run the base model over the wikitext contexts and accumulate, per layer, the statistics the template needs.

Residual = output of block `layer` at the last context token (the position that predicts the target word).
Saved: per-word sums/counts and the full second moment of the fit split; raw residuals of the held-out split.
"""
import time

from _common import LAYERS, OUT, load_base, torch

t0 = time.time()
C = torch.load(OUT / "contexts.pt")
ids, lens, widx, split = C["ids"].long(), C["lens"].long(), C["word_idx"].long(), C["split"]
N, W, nL, d = len(ids), len(C["words"]), len(LAYERS), 3584
model = load_base()
dev = model.device
pad = 0
print(f"{N} contexts, {W} words, layers {LAYERS}; model loaded in {time.time() - t0:.0f}s")

sum_w = torch.zeros(nL, W, d, dtype=torch.float64, device=dev)
cnt_w = torch.zeros(W, dtype=torch.float64, device=dev)
xx = torch.zeros(nL, d, d, dtype=torch.float64, device=dev)
held_idx = torch.nonzero(split == 1).squeeze(1)
held = torch.zeros(len(held_idx), nL, d, dtype=torch.float16)
held_pos = {int(n): k for k, n in enumerate(held_idx)}

order = torch.argsort(lens)
BS = 256
with torch.no_grad():
    for s in range(0, N, BS):
        b = order[s:s + BS]
        L = int(lens[b].max())
        x = ids[b, :L].clone()
        mask = (torch.arange(L)[None, :] < lens[b, None]).long()
        x[mask == 0] = pad
        hs = model.model(input_ids=x.to(dev), attention_mask=mask.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        last = (lens[b] - 1).to(dev)
        rows = torch.arange(len(b), device=dev)
        H = torch.stack([hs[l + 1][rows, last] for l in LAYERS], 1).float()  # [B, nL, d]
        fit = (split[b] == 0).to(dev)
        Hf, wf = H[fit], widx[b].to(dev)[fit]
        sum_w.index_add_(1, wf, Hf.transpose(0, 1).double())
        cnt_w.index_add_(0, wf, torch.ones(len(wf), dtype=torch.float64, device=dev))
        xx += torch.einsum("bld,ble->lde", Hf.double(), Hf.double())
        for k in torch.nonzero(~fit).squeeze(1).tolist():
            held[held_pos[int(b[k])]] = H[k].half().cpu()
        if (s // BS) % 50 == 0:
            print(f"  {s}/{N}  {time.time() - t0:.0f}s", flush=True)

n_fit = cnt_w.sum()
mu = sum_w.sum(1) / n_fit
cov = xx / n_fit - torch.einsum("ld,le->lde", mu, mu)
wmean = sum_w / cnt_w[None, :, None].clamp(min=1)
print("fit contexts:", int(n_fit), "| min per word:", int(cnt_w.min()), "| residual norm per layer:",
      [round(float(mu[i].norm()), 1) for i in range(nL)])
torch.save({"layers": LAYERS, "words": C["words"], "token_ids": C["token_ids"], "mu": mu.float().cpu(), "cov": cov.float().cpu(),
            "wmean": wmean.float().cpu(), "counts": cnt_w.cpu(), "held": held, "held_word": widx[held_idx].int()}, OUT / "stats.pt")
print(f"saved stats.pt in {time.time() - t0:.0f}s")
