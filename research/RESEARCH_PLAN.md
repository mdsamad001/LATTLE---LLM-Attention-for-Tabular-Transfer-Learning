# Research Plan: Why LLMs Fail on Tabular Data, and What LATTLE Actually Fixes

**Status:** draft v1 — 2026-08-27
**Owner:** Ibna Kowsar
**Repo:** https://github.com/kawseribn/LATTLE

---

## 1. Motivation

The LATTLE paper (arXiv:2511.06161) claims that transplanting frozen key/value
projections from a source-fine-tuned DistilGPT2 into a gated Feature Tokenizer
Transformer (gFTT) enables cross-domain tabular transfer. A full corrected
reproduction (100 fold-runs, 10 seeds, 100 Optuna trials per fold) shows the
*empirical wins hold* but the *stated mechanism does not explain them*.

This plan turns that discrepancy into a stronger contribution: a mechanistic
account of why language models fail on tabular prediction and why a
feature-tokenized architecture recovers the loss.

---

## 2. What we established empirically

All numbers below come from the corrected pipeline (commit `db048e5`), run on
8× RTX 2080 Ti, 90 source fine-tunes + 100 transfer fold-runs, 0 failures.
Raw data in `results_step1_raw.csv`, `results_step3_raw.csv`.

### F1. Transfer performance is set by the target, not the source

Within-target spread across *different* sources, versus spread *between*
targets:

| Target | spread across sources | source AUC range |
|---|---|---|
| cylinder-bands | 0.004 | 0.764 – 1.000 |
| vehicle | 0.006 | 0.958 – 0.998 |
| cmc | 0.007 | 0.894 – 0.976 |
| Diabetes | 0.012 | 0.743 – 0.800 |
| pc1 | 0.020 | 0.998 – 1.000 |
| **between targets** | **0.242** | — |

A source that learned nothing (seismic_bumps, majority-class collapse, AUC
0.764) transfers as well as one that solved its task by leakage
(steel-plate-fault, AUC 1.000).

### F2. Architecture and tuning are nearly irrelevant

Changing 2→5 gFTT layers, FFN 256→2048, moving the transplant from the top
layer to l0 (as the paper specifies), fixing inert dropout, and going from 15
to 100 Optuna trials moved mean AUC by **−0.001** (0.8238 → 0.8227).

### F3. Random K/V ≈ fine-tuned K/V (weak, n=1)

The original code silently transplanted *randomly initialised* weights (a
checkpoint key-prefix bug). At matched training budget it scored 0.748 against
0.752 for the corrected pipeline — indistinguishable. Single fold, single
trial; suggestive only.

### F4. Our corrected numbers sit below the published ones

Mean AUC 0.826 vs 0.852 across the same 10 pairs; 8/10 pairs lower;
Wilcoxon signed-rank **p = 0.0098**. Leading explanation: the original
`evaluate_final` used the test set as the validation set with `save_best=True`,
i.e. checkpoint selection on test data. We validate on a held-out split.

### F5. Data-quality issues affecting 4 of 10 pairs

- **steel-plate-fault**: target leakage. Columns V28–V33 are one-hot fault
  flags whose row-sum is 1 for every Class=1 row and 0 for every Class=2 row.
  Affects both steel-plate-fault pairs.
- **seismic_bumps**: majority-class collapse on all 10 seeds (F1 std 0.001).
- **pc1 targets**: F1 ≈ 0.29 against accuracy ≈ 0.93 — majority-class
  prediction that AUC-only reporting hides.

---

## 3. Central hypothesis

> LLMs fail at tabular prediction because flat text serialization collapses
> under feature dimensionality. LATTLE's gain comes from moving computation
> into a **feature-tokenized architecture** that does not suffer this collapse
> — not from the transferred weights.

This is consistent with every finding above (F1–F3), with the paper's own
Table 2 (LLM cross-domain ranks 9th; LATTLE, same LLM, ranks 1st), and with
[Why LLMs Fail at Tabular Prediction](https://arxiv.org/abs/2608.02412), which
identifies input dimensionality as the sole driver of LLM tabular collapse.

---

## 4. Research questions

- **RQ1** Does the transplant contribute at all, and does the *source* of the
  K/V matter? (causal)
- **RQ2** Does the transplant's contribution depend on **target sample size**
  — i.e. is it a data-efficiency prior tested in the wrong regime?
- **RQ3** What breaks *inside an LLM* as tabular feature dimensionality grows?
  (mechanistic)
- **RQ4** Does a feature-tokenized architecture avoid that failure mode, and
  can we show it at matched dimensionality?

---

## 5. Work packages

### WP0 — Evaluation hygiene (no new compute)
Removes reviewer attack surface. Uses data already on disk.

| Task | Output |
|---|---|
| Wilcoxon signed-rank / Friedman + CD diagram vs baselines | fills checklist item 4.12, currently "no" |
| Report AUC **and** F1 **and** accuracy everywhere | defuses the pc1 majority-class problem |
| Re-run steel-plate-fault without V28–V33 (or OpenML 40982) | removes leakage from 2 pairs |
| State deviations: `eval_auc` selection, early stopping patience 10, cosine scheduler | protocol transparency |

**Cost:** ~1 day, most of it no GPU. **Do this regardless of everything else.**

### WP1 — Causal ablation ladder (decides the framing)

| Arm | K/V source | Question |
|---|---|---|
| A | source-fine-tuned DistilGPT2 | current method |
| B | stock DistilGPT2, no fine-tuning | does source data matter? |
| C | random, **magnitude-matched** (same per-matrix mean/std) | does language structure matter? |
| D | no transplant | does the mechanism help at all? |
| E | K/V from a non-language transformer (ViT) | is it *language* or just *pretrained* structure? |

Plus **l0 knockout**: ablate layer 0 entirely; if l1–l4 compensate, the
transplant site is not load-bearing.

**Method:** reuse the tuned hyperparameters already stored in the 96 Optuna
SQLite studies (`transfer_learning_results_w*/optuna_studies/*.db`). One
training per (arm, pair, fold) instead of 101 — a 100× saving.

**Cost:** ~3 h for arms B/C/D on 8 GPUs. **Success criterion:** any arm
differing from A by more than the fold std (≈0.03 AUC).

**Decision gate:**
- `A ≈ B > C` → paper's "general language bridge" claim is *vindicated*; write it up as a demonstrated result
- `A ≈ B ≈ C ≈ D` → transplant is inert; pivot to the architectural claim (Section 3)
- `A > B` → source fine-tuning matters after all; the original framing survives

### WP2 — Low-resource sweep (highest chance of rescuing the mechanism)

The paper motivates "practical low-resource learning scenarios" but trains on
**70% of every target**. With 1,000+ target samples and 150 epochs, any
initialization washes out — which would explain F1, F2, and F3 in one stroke.

**Design:** sweep target training size n ∈ {25, 50, 100, 200, 500, full} ×
{arm A, arm D} × 10 pairs × 5 folds.
**Prediction:** the A−D gap grows monotonically as n shrinks.
**Cost:** ~1 day. **Payoff:** if the gap opens, the mechanism is real and the
evaluation protocol was simply hiding it — a stronger and better-motivated
claim than the current one.

### WP3 — K/V subspace decomposition (zero GPU)

We hold 90 K/V matrices (9 sources × 10 seeds). Stack and decompose into a
shared subspace plus source-idiosyncratic residuals; transplant each
separately.

| Transplant | Tests |
|---|---|
| shared component only | is the bridge generic? |
| residual only | does source identity carry anything? |
| full | baseline |

Turns the paper's Section 5 assertion into a measurement. **Cost:** hours,
pure analysis on existing artifacts.

### WP4 — Mechanistic study of dimensionality collapse (the new contribution)

Addresses RQ3/RQ4. Builds on 2608.02412, which establishes *that*
dimensionality drives collapse but not *what breaks*.

1. **Dimensionality sweep, matched:** same data, same target, vary feature
   count d ∈ {2, 5, 10, 20, 50}; compare serialized-LLM vs gFTT accuracy.
   Establishes the architectural claim behaviourally.
2. **SAE on the LLM residual stream** while it reads serialized rows. Look for
   features encoding *similarity to in-context examples* (the local
   distance-based classifier behaviour 2608.02412 reports). Track whether they
   fragment, saturate, or get crowded out as d grows.
3. **Causal confirmation:** activation patching / knockout on the identified
   features — attention selectivity alone is not evidence
   ([TabPFN head study](https://arxiv.org/html/2606.12917) found a head with
   entropy 0.22 and *near-zero* causal necessity).

**Why SAEs are safe here:** we would be running them on a *language model*,
where superposition demonstrably holds and tooling is mature — not on a
tabular model, where
[Superposition Is Not Necessary](https://arxiv.org/pdf/2605.05151) (time
series, the closest modality) casts doubt on the premise. Use the `saelens`
skill.

**Cost:** weeks. **Do only after WP1/WP2 settle the framing.**

---

## 6. Related work to position against

**Must cite — closest prior art, currently missing:**
[Lu et al., *Pretrained Transformers as Universal Computation Engines*](https://arxiv.org/pdf/2103.05247)
(AAAI 2022). Freezes GPT-2 self-attention, transfers to vision/protein/numeric
tasks, tuning 0.1% of parameters and none of self-attention. **Already ran our
arm C and arm E**: frozen pretrained beats frozen random by +2.5 to +24.2
points, and pretraining modality matters (language > bit-memory > random; ViT
mixed). Their conclusion — LM self-attention has properties amenable to
universal computation — is essentially LATTLE's Section 5 claim.

*Implication:* our novelty must be stated as **selective K/V transplant into a
heterogeneous architecture under disjoint feature spaces**, which FPT does not
cover. If arm C shows no gap where FPT found one, that difference is itself the
finding and needs explaining.

**Tabular mechanistic interpretability (2026 wave):**
- [A Mechanistic Study of Tabular Foundation Models](https://arxiv.org/html/2605.21288v1) — probes, knockout, causal interventions on TabPFNv2 / TabICLv2 / Mitra
- [Where Computation Lives Inside TabPFN](https://arxiv.org/html/2606.12917) — activation patching, attention entropy, head localization
- [Is One Layer Enough?](https://arxiv.org/html/2605.06510v1)
- [In Search of Grandmother Cells](https://arxiv.org/pdf/2601.03657) — only *moderate* neuron selectivity in tabular representations

**LLM ↔ tabular:**
- [Why LLMs Fail at Tabular Prediction](https://arxiv.org/abs/2608.02412) — dimensionality is the driver; falsifies tokenization/serialization/class-overlap explanations
- [LLM Embeddings for Deep Learning on Tabular Data](https://arxiv.org/pdf/2502.11596)
- [Instability of LLM text embeddings for tabular data](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2026.1851023/full) — feature-order sensitivity
- [Robustness limitations of LLMs on tabular data](https://academic.oup.com/pnasnexus/article/5/6/pgag197/8699520)

**SAE prior art on tabular (already taken):**
[XNNTab](https://arxiv.org/pdf/2512.13442), [Towards Interpretable DNNs for Tabular Data](https://arxiv.org/pdf/2509.08617)

**Number representation (do not pitch as novel):**
[digit-wise base-10](https://arxiv.org/html/2410.11781),
[logarithmic compression](https://arxiv.org/pdf/2602.07812),
[Weber's law geometry](https://arxiv.org/pdf/2603.20642)

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| All ablation arms tie → mechanism claim dies | **high**, given F1–F3 | Pivot to the architectural claim (Section 3); frame frozen K/V as a structured prior, citing FPT and random-features literature |
| Superposition does not hold → SAE returns meaningless dictionaries | medium for tabular models, **low for LLMs** | WP4 runs SAEs on the LLM only; test the premise (participation ratio, polysemanticity) before building |
| FPT judged as prior art that subsumes the contribution | medium | Position on selective K/V + heterogeneous architecture + disjoint features; cite explicitly rather than being caught |
| Low-resource gap fails to appear | medium | Still a publishable negative that completes the story |
| Reviewers attack pc1 F1 / steel-plate-fault leakage | **high if unfixed** | WP0 |

---

## 8. Sequencing

```
WP0 hygiene        ▓▓                      (1 day, no GPU, do now)
WP3 decomposition   ▓                      (hours, no GPU)
WP1 ablation ladder  ▓▓▓                   (~3 h GPU)   ← decision gate
WP2 low-resource       ▓▓▓▓▓               (~1 day GPU)
WP4 mechanistic              ▓▓▓▓▓▓▓▓▓▓    (weeks, gated on WP1/WP2)
```

**Immediate next actions:** WP0 and WP3 need no GPU time and no new runs.
WP1 needs ~3 hours and settles which paper we are writing.

---

## 9. Candidate framings by outcome

| WP1/WP2 outcome | Paper |
|---|---|
| Arm C < A, low-resource gap opens | *"LLM attention priors improve low-resource cross-domain tabular transfer"* — original claim, now demonstrated rather than asserted |
| All arms tie, gFTT beats serialized LLM at matched d | *"LLMs fail at tabular prediction under feature dimensionality; feature-tokenized architectures recover it"* — mechanistic, uses all findings, explains the nulls |
| Mixed | Report both; the honest combination is still novel |
