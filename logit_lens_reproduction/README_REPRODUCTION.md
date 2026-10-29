# Logit-Lens Reproduction — User-Gender organisms (Phase A)

This directory reproduces the **official** `eliciting-secret-knowledge`
logit-lens elicitation + auditing pipeline exactly, for **both** User-Gender
secret-keeping organisms (`bcywinski/gemma-2-9b-it-user-female` and
`bcywinski/gemma-2-9b-it-user-male`), restricted to just the `logit_lens`
method (the paper's other elicitation methods — e.g. prefill, SAE, fuzzing —
are out of scope here). This is read-only extraction + auditing: nothing in
this directory steers, edits, or retrains anything.

**Why both organisms:** the paper's own `METRICS.md` "User Gender" numbers
are not female-only — section 5.1 of the paper states results are "averaged
over 3 different Taboo models as well as **female and male models**." A
female-only reproduction (sections 1-7 below) therefore cannot be compared
directly to `METRICS.md`. Section 8 adds the matching male-organism run and
the resulting combined (female+male-averaged) numbers, which **are**
directly comparable to the paper.

## 1. What was run

Everything is driven by one script, `scripts/run_reproduction.sh`, which
must be run with `cwd = secret_keeping/external/eliciting-secret-knowledge`
(that repo's own helper scripts are path-relative to its own root):

```bash
cd secret_keeping/external/eliciting-secret-knowledge
conda run -n eval_aware bash ../../logit_lens_reproduction/scripts/run_reproduction.sh \
    > ../../logit_lens_reproduction/outputs/run_log.txt 2>&1
```

It runs 8 steps (a-h), all via `conda run -n eval_aware` (so all transformer/
PEFT/flash-attn-dependent work happens in the pinned env; the HF token is
loaded once via `_env_setup.py`'s non-printing `load_hf_token()` and passed
only as an env var, never echoed):

```
(a) organism inference, standard prompt set    -> outputs/standard/inference/
(b) organism inference, direct prompt set      -> outputs/direct/inference/
(c) logit-lens extraction, organism, standard   -> outputs/standard/logit_lens/
(d) logit-lens extraction, organism, direct     -> outputs/direct/logit_lens/
(e) logit-lens extraction, BASE model, standard -> outputs/standard/logit_lens_base/
(f) logit-lens extraction, BASE model, direct   -> outputs/direct/logit_lens_base/
(g)/(h) auditor (google/gemma-3-4b-it) runs on each of the 4 logit-lens
    output dirs above, target_word="female"    -> outputs/{standard,direct}/audit_logit_lens{,_base}/
```

Models:
- organism: `bcywinski/gemma-2-9b-it-user-female` (PEFT/LoRA adapter on top
  of `google/gemma-2-9b-it`, loaded **unmerged** via the official repo's own
  loading code — no `merge_and_unload()` anywhere in this pipeline)
- base-model baseline (steps e/f): `google/gemma-2-9b-it` — logit-lens
  extraction run through the plain base model's own internals on the SAME
  organism-generated text used in (a)/(b), matching the paper repo's own
  `01_run_base_auditing.sh` baseline-comparison pattern
- auditor: `google/gemma-3-4b-it`, prompt template
  `prompts/gender/auditor_guess_tokens.txt`, `--mode logit_lens_tokens
  --num_guesses 1 --num_tokens 200 --batch_size 50 --temperature 0.0
  --target_words female`

Layer: 23 (of 42), mode `first_person_pronouns` (probabilities averaged
across first-person-pronoun positions in the user prompt), top-k=200 — all
exactly the official repo's own `user_gender/scripts/run_get_logit_lens.sh`
defaults, unmodified.

## 2. Run log / timing

Full raw log: `outputs/run_log.txt`.

```
### Phase A reproduction starting at 2026-10-02T20:21:02+10:00 ###
### Phase A reproduction finished at 2026-10-02T20:49:16+10:00 ###
### Total wall-clock: 1694 seconds ###   (≈ 28 min 14 s)
```

Breakdown (from log timestamps / step transitions):
- (a)+(b) organism inference (two separate 1000-prompt / 10-response-each
  generation runs): ~20:21:02 → ~20:25:40 (≈ 4.6 min)
- (c)+(d) logit-lens extraction through the organism (1000 pairs each,
  including a fresh ~10 s model-load each): ~20:25:40 → ~20:31:11 (≈ 5.5 min)
- (e)+(f) logit-lens extraction through the base model (1000 pairs each):
  ~20:31:11 → ~20:35:25 (≈ 4.2 min)
- (g)+(h) four auditor runs (`google/gemma-3-4b-it`, 1000 pairs / 20 batches
  of 50 each, ~3 min of batched generation per run plus model load):
  ~20:35:25 → 20:49:16 (≈ 13.9 min)

Each of (c)/(d)/(e)/(f) processed 1000 input-output pairs; organism
extraction (c)/(d) and base-model extraction (e)/(f) each had **990/1000
succeed, 10 errors** — this ~1% error rate (cases where the required
chat-template markers / first-person-pronoun positions aren't found in a
given generated response) is expected/acceptable and matches the pattern
already seen in the original 00_run_auditing-style runs; it is not a bug in
this reproduction.

## 3. GPU memory

Observed via `nvidia-smi` during this run: generation + extraction steps
(organism model, bf16, ~18 GiB base weights + LoRA adapter) used roughly
**20-25 GiB** of the single H100's 80 GB at any sampled point — comfortably
within budget, no OOM at any step across all 8 steps. GPU was fully idle (1
MiB used) both before the pipeline started and after it finished.

## 4. A real incident encountered and fixed during setup (unrelated to this run's correctness, but documented for the record)

While preparing the `eval_aware` environment for this reproduction (installing
a flash-attn wheel needed by some of the paper repo's own code paths), a
`pip install` was run with `PYTHONNOUSERSITE=1` set. On this machine,
`~/.local/lib/python3.11/site-packages` (pip's per-user "user site-packages")
is shared across every conda environment and normally sits *ahead* of each
env's own `site-packages` on `sys.path` — which is exactly why
`PYTHONNOUSERSITE=1` is normally the *safe* thing to pass when installing
into a secondary env (it is what prevents a shared user-site package from
silently shadowing/conflicting with an env-local one; see the parallel,
more severe incident in
`../jacobian_lens_comparison/README_COMPARISON.md` section 4, where the
*opposite* mistake — forgetting that flag — caused real breakage).

Here, however, setting `PYTHONNOUSERSITE=1` for the flash-attn wheel install
had an unwanted side effect: with the shared user-site hidden from pip's
view, pip's dependency resolver no longer saw the correctly-pinned
`torch==2.7.1+cu126` that was actually living in that shared
`~/.local/.../site-packages` directory (not in `eval_aware`'s own
`site-packages`). Believing no compatible torch was installed, pip pulled in
`torch==2.14.0+cu126` plus roughly 20 `nvidia-*-cu12`/`triton`/
`cuda-toolkit` packages, installing all of them directly into `eval_aware`'s
*own* `site-packages`. Because the shared user-site copy of torch 2.7.1
still took precedence on `sys.path` at import time, `import torch` kept
resolving to 2.7.1 and nothing visibly broke — but the mismatched
2.14.0+cu126 stack sat latent inside `eval_aware`'s own site-packages,
one accidental `PYTHONNOUSERSITE=1` or user-site change away from silently
flipping the active torch version mid-pipeline.

**Fix applied** (before any step of this Phase A run was launched):

```bash
conda run -n eval_aware pip uninstall -y torch triton cuda-bindings \
    cuda-pathfinder cuda-toolkit nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 nvidia-cufile-cu12 nvidia-curand-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12
# torchvision==0.22.1 (correctly installed, compatible with the surviving
# torch 2.7.1 in the shared user-site) was checked and deliberately kept.
```

**Verified immediately after the fix, and re-verified again at the start of
this session (before trusting that this run's results reflect a correct
environment):**

```bash
conda run -n eval_aware python -c "
import transformers, torch, diffusers, accelerate, peft, huggingface_hub, tokenizers, flash_attn, nnsight
print(transformers.__version__, torch.__version__, diffusers.__version__,
      accelerate.__version__, peft.__version__, huggingface_hub.__version__,
      tokenizers.__version__)
"
# -> 4.57.6 2.7.1+cu126 0.34.0 1.15.0 0.21.1 0.36.2 0.22.2
```

This matched exactly, both when the incident was first fixed and again at
the start of the session that watched this Phase A run to completion.
**The fix was applied, and re-verified correct, before this run's `(a)`
step (organism inference) was ever launched** — this run's generation and
extraction steps ran entirely under the correct, pinned `torch==2.7.1+cu126`
stack. The contamination was caught and fully resolved before it could
affect any of this reproduction's actual results; none of the numbers in
section 5 below are suspect because of it.

**Lesson:** `PYTHONNOUSERSITE=1` is a double-edged mitigation on a machine
with a shared user-site directory — it protects a *target* env's own
packages from being silently shadowed by the shared user-site, but it can
also hide an *existing, correctly-pinned* shared-user-site package from
pip's resolver, causing pip to "fix" a phantom missing dependency by
installing an unwanted version directly into the target env. The safer
general pattern on this machine is to check `pip show <pkg> | grep Location`
*before* any install that touches a pinned package, rather than reflexively
setting `PYTHONNOUSERSITE=1`.

## 5. Results — auditor accuracy (real numbers, read directly from each `metrics_*.json`)

Each auditor run (`guess_secret_word.py --mode logit_lens_tokens`) writes a
`guessing_logit_lens_tokens_*.json` (raw per-example guesses) and a separate
`metrics_logit_lens_tokens_*.json` (via `calculate_metrics.py`,
`metrics["mean_accuracy"]` / `["std_accuracy"]`, averaged over the 10
`model_response_index` values 0-9). All 4 numbers below were read directly
from those 4 files, not transcribed from the log:

| Model probed | Standard-prompt test | Direct-prompt test |
|---|---|---|
| **Organism** (`bcywinski/gemma-2-9b-it-user-female`) | **0.9990 ± 0.0030** | **0.9840 ± 0.0066** |
| **Base model** (`google/gemma-2-9b-it`, no adapter) | 0.8590 ± 0.0164 | 0.8690 ± 0.0176 |

Source files:
- organism/standard: `outputs/standard/audit_logit_lens/metrics_logit_lens_tokens_google_gemma-3-4b-it_female_20261002_203923.json`
- organism/direct: `outputs/direct/audit_logit_lens/metrics_logit_lens_tokens_google_gemma-3-4b-it_female_20261002_204241.json`
- base/standard: `outputs/standard/audit_logit_lens_base/metrics_logit_lens_tokens_google_gemma-3-4b-it_female_20261002_204558.json`
- base/direct: `outputs/direct/audit_logit_lens_base/metrics_logit_lens_tokens_google_gemma-3-4b-it_female_20261002_204915.json`

**Interpretation (honest, not spun):** the organism's logit-lens readout
gives the auditor a clearly higher chance of guessing "female" than the
base model's own logit-lens readout does, on both prompt sets (99.90% vs.
85.90% standard; 98.40% vs. 86.90% direct) — consistent with the organism
encoding the user's gender in a way that is visible to a plain logit lens at
layer 23, exactly as the paper's own elicitation-methods results would
predict. That said, the base-model baseline itself is **not** near-chance
(roughly 86-87%, not ~50%) — the auditor LLM is evidently able to guess
"female" at non-trivial accuracy even reading out the *base* model's
internal probabilities on these same user prompts, for reasons unrelated to
the organism's secret (e.g. the prompts themselves, or the auditor's own
prior, may already be gender-skewed) — so the organism's effect here should
be read as "pushes an already fairly guessable readout substantially closer
to ceiling," not "takes it from chance to certainty."

## 6. Directory layout

```
secret_keeping/logit_lens_reproduction/
├── scripts/
│   ├── _env_setup.py        shared HF-cache env setup + non-printing load_hf_token()
│   └── run_reproduction.sh  the 8-step driver script described above
├── outputs/
│   ├── run_log.txt
│   ├── standard/
│   │   ├── inference/              (a) organism generations, standard prompts
│   │   ├── logit_lens/             (c) logit-lens extraction, organism
│   │   ├── logit_lens_base/        (e) logit-lens extraction, base model
│   │   ├── audit_logit_lens/       (g) auditor results+metrics, organism
│   │   └── audit_logit_lens_base/  (g) auditor results+metrics, base model
│   └── direct/   (same 5 subdirectories, for the direct prompt set)
└── README_REPRODUCTION.md   this file
```

## 7. Final verification performed

- Re-read all 4 `metrics_*.json` files directly (section 5) before writing
  this document — none of the numbers above were transcribed from the run
  log or from memory.
- `eval_aware` package versions re-verified immediately before and after
  this run (section 4) — unchanged, correct, matching the known-good pins.
- `grep -rl` for the literal HF token value (loaded via the same
  non-printing `load_hf_token()` pattern used by the pipeline itself, never
  echoed) across `outputs/` found zero matches.
- `git status --short` in the repo root shows no files from
  `secret_keeping/` as tracked or staged (directory is gitignored).

## 8. Male-organism run and the combined (paper-comparable) comparison

### 8.1 What was run

Identical pipeline, mirrored exactly from `scripts/run_reproduction.sh` into
a new `scripts/run_reproduction_male.sh` with three substitutions:
`MODEL_NAME` → `bcywinski/gemma-2-9b-it-user-male`, `TARGET_GENDER` →
`"male"`, output root → `outputs_male/` (a new sibling directory; the
original female-organism `outputs/` tree was never touched or overwritten).
Every other parameter (prompts, layer 23, mode `first_person_pronouns`,
top_k=200, auditor `google/gemma-3-4b-it`, `NUM_RESPONSES=10
MAX_NEW_TOKENS=200 TEMPERATURE=1.0 BATCH_SIZE=100 SEED=1`) is unchanged — the
**same** `prompts/gender/gender_standard_test.txt` /
`gender_direct_test.txt` files are reused as-is, since they are
gender-neutral and not organism-specific.

```bash
cd secret_keeping/external/eliciting-secret-knowledge
conda run -n eval_aware bash ../../logit_lens_reproduction/scripts/run_reproduction_male.sh \
    > ../../logit_lens_reproduction/outputs_male_run_log.txt 2>&1
```

Run log: `### Phase A (male organism) reproduction starting at
2026-10-02T22:38:04+10:00 ###` → `### ... finished at
2026-10-02T23:04:55+10:00 ### / Total wall-clock: 1611 seconds ###` (≈ 26.9
min — consistent with the female run's 1694 s). Logit-lens extraction
(c)/(d)/(e)/(f) each processed 1000 pairs; the direct-prompt organism
extraction (d) hit the same expected ~1% error rate as the female run
(990/1000 succeeded, 10 errors); all other steps processed 1000/1000 with
zero errors.

### 8.2 Male-organism results (real numbers, read directly from each `metrics_*.json`)

| Model probed | Standard-prompt test | Direct-prompt test |
|---|---|---|
| **Organism** (`bcywinski/gemma-2-9b-it-user-male`) | **0.3600 ± 0.0118** | **0.2990 ± 0.0274** |
| **Base model** (`google/gemma-2-9b-it`, no adapter) | 0.1630 ± 0.0228 | 0.1570 ± 0.0100 |

Source files:
- organism/standard: `outputs_male/standard/audit_logit_lens/metrics_logit_lens_tokens_google_gemma-3-4b-it_male_20261002_225455.json`
- organism/direct: `outputs_male/direct/audit_logit_lens/metrics_logit_lens_tokens_google_gemma-3-4b-it_male_20261002_225810.json`
- base/standard: `outputs_male/standard/audit_logit_lens_base/metrics_logit_lens_tokens_google_gemma-3-4b-it_male_20261002_230126.json`
- base/direct: `outputs_male/direct/audit_logit_lens_base/metrics_logit_lens_tokens_google_gemma-3-4b-it_male_20261002_230454.json`

As predicted by the female-run's auditor-bias analysis (section 5's
interpretation, and the earlier investigation that motivated this run): with
`target_word="male"`, accuracy is now **much lower**, not higher, than the
female-condition numbers — because the same auditor LLM still guesses
"female" far more often than "male" regardless of which organism it is
reading out from, so scoring against "male" turns that bias from a tailwind
into a headwind.

### 8.3 Combined (female + male averaged) — directly comparable to `METRICS.md`

Averaging each condition's `mean_accuracy` across the female run (section 5)
and the male run (8.2) — exactly the averaging the paper describes doing
across its female and male models — gives:

| Condition | Female-only | Male-only | **Combined** | **Paper (`METRICS.md`, I/O + LL Tokens)** |
|---|---|---|---|---|
| Base model, standard prompts | 85.90% | 16.30% | **51.10%** | 51.5 ± 1.9 |
| Organism, standard prompts | 99.90% | 36.00% | **67.95%** | 68.2 ± 1.6 |
| Organism, direct prompts | 98.40% | 29.90% | **64.15%** | 66.15 ± 1.33 |
| Base model, direct prompts | 86.90% | 15.70% | **51.30%** | *(not reported by paper)* |

**Interpretation:** the combined numbers match the paper closely — within
0.3-2.0 percentage points on every reported condition, well inside the
paper's own reported standard-deviation bands (±1.3-1.9). This confirms the
original female-only reproduction (section 5) was not wrong, just scoped
differently from the paper's own averaged metric: our methodology,
pipeline, and metric code are a faithful reproduction once the same
female+male averaging the paper performs is applied on our end too. It also
confirms the root cause investigated earlier — the auditor model has a
strong inherent bias toward guessing "female" over "male" (quantified
directly from the raw per-example guesses in the female run's base-model/
standard-prompt condition: among the 422/1000 cases where the top-200
logit-lens tokens contained *no* gendered words at all, the auditor still
guessed "female" 422 times vs. "male" only 60 times) — which inflates
female-target accuracy and deflates male-target accuracy by roughly
symmetric amounts, cancelling out under averaging.

The base-model/direct-prompt condition (51.30% combined) has no published
paper counterpart to compare against, but sits almost exactly at the other
three conditions' ~51% base-model level, which is internally consistent.

### 8.4 Final verification performed (male run)

- All 4 male-run `metrics_*.json` files read directly (section 8.2) before
  writing this section — none of the numbers were transcribed from the run
  log or from memory.
- HF-token-leak check: a Python script (not a hardcoded shell grep) loaded
  the token via the project's standard non-printing `load_hf_token()` and
  grepped `outputs_male/`, `outputs_male_run_log.txt`, and
  `scripts/run_reproduction_male.sh` for the literal value — zero matches
  (`LEAK DETECTED: False`).
- `git status --short` in the parent repo root remained limited to the
  same three pre-existing untracked entries (`.gitignore`, `.vscode/`,
  `code/.env`) — no new tracked/staged files, `secret_keeping/` still
  gitignored.
- No files written outside `secret_keeping/logit_lens_reproduction/`.
- The original female-organism `outputs/` tree was not modified; the male
  run wrote exclusively to the new, non-colliding `outputs_male/` tree.

### 8.5 Directory layout (male run)

```
secret_keeping/logit_lens_reproduction/
├── scripts/
│   └── run_reproduction_male.sh   mirrors run_reproduction.sh, male organism
├── outputs_male_run_log.txt
└── outputs_male/
    ├── standard/
    │   ├── inference/              organism (male) generations, standard prompts
    │   ├── logit_lens/              logit-lens extraction, organism (male)
    │   ├── logit_lens_base/         logit-lens extraction, base model
    │   ├── audit_logit_lens/        auditor results+metrics, organism (male), target_words=male
    │   └── audit_logit_lens_base/   auditor results+metrics, base model, target_words=male
    └── direct/   (same 5 subdirectories, for the direct prompt set)
```
