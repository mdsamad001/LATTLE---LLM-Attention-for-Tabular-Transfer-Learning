# Experiment Log

One row per experiment. Fill in **before** running (hypothesis, prediction) so
the result cannot be rationalised after the fact.

| ID | WP | Date | Hypothesis | Prediction if true | Config | Result | Verdict |
|---|---|---|---|---|---|---|---|
| E00 | — | 2026-08-22→27 | corrected pipeline reproduces the paper | mean AUC ≈ 0.852 | 10 seeds, 10 folds, 100 trials, 5-layer/2048 gFTT, l0 transplant | mean AUC **0.826**, 8/10 pairs below, Wilcoxon p=0.0098 | ✗ does not reproduce |
| E01 | WP1 | | random magnitude-matched K/V underperform LLM K/V | arm C < arm A by > 0.03 AUC | reuse tuned HP from optuna DBs, 10 pairs × 10 folds | | |
| E02 | WP1 | | stock (un-fine-tuned) K/V ≈ source-fine-tuned K/V | arm B ≈ arm A | as above | | |
| E03 | WP1 | | the transplant helps at all | arm D < arm A | as above | | |
| E04 | WP1 | | l0 is load-bearing | knockout drops AUC | ablate layer 0 | | |
| E05 | WP2 | | transplant is a data-efficiency prior | A−D gap grows as n shrinks | n ∈ {25,50,100,200,500,full} | | |
| E06 | WP3 | | K/V bridge is generic, not source-specific | shared subspace ≈ full; residual ≈ none | SVD over 90 K/V matrices | | |
| E07 | WP4 | | gFTT resists dimensionality collapse | LLM accuracy falls with d, gFTT does not | matched d ∈ {2,5,10,20,50} | | |

## Notes on rigour

- **Magnitude matching (E01):** random K/V must match the per-matrix mean and
  std of the DistilGPT2 weights they replace. Otherwise a null result may only
  mean "we broke the scale".
- **Hyperparameter reuse (E01–E03):** tuned on arm A, which biases *toward* A.
  If arms still tie, the null is strengthened, not weakened. State this.
- **Attention maps are not evidence (E04, E07):** confirm causally
  (patching/knockout). A TabPFN head with attention entropy 0.22 had near-zero
  causal necessity.
- **Fold variance:** per-pair std ranges 0.009–0.066 AUC. Any claimed effect
  must exceed it. Use paired tests across folds, not raw means.
