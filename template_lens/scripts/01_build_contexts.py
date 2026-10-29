#!/usr/bin/env python3
"""Build next-word contexts from wikitext-103 for the template lens.

For every vocabulary word w (one Gemma token written with a leading space) collect text windows that end right
before w and do not contain w. The residual at the last context token is what the template for w is fitted on.

Needs network on the first run:  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python 01_build_contexts.py
"""
import json
import random
import re

import numpy as np
from _common import OUT, HF, load_tokenizer, torch

from datasets import load_dataset

N_WORDS, N_PARAS = 1000, 400_000
FIT, HELD = 100, 20
MAX_CTX, MIN_CTX = 64, 8
FORCE = ["gold", "flag", "moon", "female", "male", "woman", "man", "girl", "boy"]


def detok(s: str) -> str:
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[$]) ", r"\1", s)
    return s.replace(" 's", "'s").replace(" n't", "n't").strip()


tok = load_tokenizer()
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", cache_dir=str(HF / "datasets"))
print("wikitext-103 train rows:", len(ds))

rng = np.random.default_rng(0)
idx = np.sort(rng.choice(len(ds), size=min(2 * N_PARAS, len(ds)), replace=False))
texts = [t for t in ds.select(idx)["text"] if len(t) > 200 and not t.lstrip().startswith("=")][:N_PARAS]
texts = [detok(t) for t in texts]
print("paragraphs kept:", len(texts))

para_ids = []
for s in range(0, len(texts), 2000):
    para_ids += [np.asarray(x, dtype=np.int32) for x in tok(texts[s:s + 2000], add_special_tokens=False)["input_ids"]]
print("tokens:", sum(map(len, para_ids)))

V = len(tok)
counts = np.bincount(np.concatenate(para_ids), minlength=V)
toks = tok.convert_ids_to_tokens(list(range(V)))
cand = [i for i, t in enumerate(toks) if t.startswith("▁") and len(t) > 3 and t[1:].isascii() and t[1:].isalpha() and t[1:].islower()]
cand.sort(key=lambda i: -counts[i])
vocab_ids = cand[:N_WORDS]
for w in FORCE:
    tid = tok.convert_tokens_to_ids("▁" + w)
    if tid not in vocab_ids:
        vocab_ids[-1 - FORCE.index(w)] = tid
words = [toks[i][1:] for i in vocab_ids]
W = len(words)
pos_of = {tid: j for j, tid in enumerate(vocab_ids)}
is_vocab = np.zeros(V, bool)
is_vocab[vocab_ids] = True
print(f"vocabulary: {W} words, rarest corpus count {min(counts[i] for i in vocab_ids)}; sample:", words[:8], "...",
      {w: int(counts[tok.convert_tokens_to_ids('▁' + w)]) for w in FORCE})

need = FIT + HELD
got = np.zeros(W, int)
rows, widx, rejected = [], [], 0
order = list(range(len(para_ids)))
random.Random(0).shuffle(order)
for p in order:
    ids = para_ids[p]
    for i in np.nonzero(is_vocab[ids])[0]:
        if i < MIN_CTX:
            continue
        j = pos_of[int(ids[i])]
        if got[j] >= need:
            continue
        win = ids[max(0, i - MAX_CTX):i]
        if (win == ids[i]).any() or re.search(r"\b" + words[j] + r"\b", tok.decode(win).lower()):
            rejected += 1
            continue
        rows.append(win)
        widx.append(j)
        got[j] += 1
    if (got >= need).all():
        break
print(f"contexts: {len(rows)} (rejected {rejected} that contained the word); words below quota: {int((got < need).sum())}")

N = len(rows)
ids_t = torch.zeros(N, MAX_CTX + 1, dtype=torch.int32)
lens = torch.zeros(N, dtype=torch.int32)
for n, win in enumerate(rows):
    ids_t[n, 0] = tok.bos_token_id
    ids_t[n, 1:1 + len(win)] = torch.from_numpy(win)
    lens[n] = len(win) + 1
widx_t = torch.tensor(widx, dtype=torch.int32)
split = torch.zeros(N, dtype=torch.int32)  # 0 = fit, 1 = held-out (first HELD contexts of every word)
seen = np.zeros(W, int)
for n, j in enumerate(widx):
    if seen[j] < HELD:
        split[n] = 1
    seen[j] += 1

torch.save({"words": words, "token_ids": torch.tensor(vocab_ids), "ids": ids_t, "lens": lens, "word_idx": widx_t,
            "split": split}, OUT / "contexts.pt")
sample = {w: [tok.decode(ids_t[n, :lens[n]]) for n in np.nonzero(np.asarray(widx) == words.index(w))[0][:3]] for w in FORCE}
json.dump(sample, open(OUT / "contexts_sample.json", "w"), indent=1)
print("saved", OUT / "contexts.pt", "| fit:", int((split == 0).sum()), "held-out:", int((split == 1).sum()))
for w in ["moon", "female"]:
    print(f"\n[{w}]")
    for c in sample[w][:2]:
        print("  ...", c[-160:].replace("\n", " "), f"[{w}]")
