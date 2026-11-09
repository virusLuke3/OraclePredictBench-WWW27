# Executed mathematical specification

This document describes the implementation actually run in this repository. The
two `*_design_reference.md` files preserve the supplied design guidance; they are
not execution evidence. Symbol names and method IDs here match the result tables.

## Target and inputs

For request $i$, $T_i$ is its proposal event time and $S_i=1$ denotes membership
in the retrospectively challenged, linked, settled, transfer-reconciled cohort.
The outcome is $Y_i=\mathbb1\{\mathrm{int}(v_i^{resolved})\ne\mathrm{int}(v_i^{proposed})\}$, including special integer resolution states.
The target $p_i=P(Y_i=1\mid X_i(T_i),S_i=1)$ is eventual protocol disagreement.
There is no fixed forecast horizon and no independent factual-truth label.

The 11 allowed features are defined in `task_spec.yaml`. An actor's history contains
only retained-primary requests settled strictly before the current proposal in
the lexicographic order `(UTC, block, transaction index, log index)`. For actor
role $a\in\{proposer,requester\}$, let $H_{ia}$ be this set, $n_{ia}=|H_{ia}|$,

$$
d_{ia}=\frac{\sum_{j\in H_{ia}}\mathbb1\{j\text{ disputed}\}}{n_{ia}},\qquad
r_{ia}=\frac{\sum_{j\in H_{ia}}\mathbb1\{j\text{ disputed}\}Y_j}{n_{ia}}.
$$

Both rates are null when $n_{ia}=0$. Their denominator is all prior settled
primary requests, not only challenged requests. The current task's cohort
conditioning is different; treating these rates as calibrated current-target
probabilities is not warranted. Each method receives the same 11 admissible
inputs. AutoGluon's training-fitted generator drops constant `liveness_hours`,
leaving 10 effective tree inputs; this is a learned preprocessing choice.

## Conventional methods

Let $\ell(y,p)=-y\log p-(1-y)\log(1-p)$ and $\sigma(z)=1/(1+e^{-z})$.

| ID | Model and prediction | Fit or selection objective | Actual configuration |
|---|---|---|---|
| Prior | $\hat p_i=\hat\pi=n_{tr}^{-1}\sum_{tr}Y_i$ | Constant Bernoulli MLE; also constant Brier minimizer on train | 282/474 = 0.5949367089 |
| LR | $\hat p_i=\sigma(\beta_0+\phi(X_i)^\top\beta)$ | $\sum_{tr}\ell(Y_i,p_i)+\|\beta\|_2^2/(2C)$; intercept unpenalized | C in {0.01,0.1,1,10}; selected 0.01; lbfgs, max_iter=3000, seed 17 |
| RF | $\hat p_i=B^{-1}\sum_{b=1}^B\hat\pi_{L_b(X_i)}$ | Greedy weighted Gini reduction $I(n)-\sum_c(n_c/n)I(n_c)$, $I(n)=2q_n(1-q_n)$ | 500 bootstrap trees; depth {4,8}, leaf minimum {3,10}; selected depth 8/leaf 10; sqrt feature sampling, seed 17 |
| GBM | $\hat p_i=\sigma(F_0+0.03\sum_{m=1}^{M}f_m(X_i))$ | Binary logistic loss, leaf-wise gradient boosting; validation Brier for iteration/candidate selection | Leaves {7,15}, leaf minimum {10,30}; max 1000 rounds; selected 7/10 with 36 fitted rounds |
| Ensemble | $\hat p_i=\sum_mw_m\hat p_{im},\ w_m\ge0,\sum_mw_m=1$ | Greedy validation ensemble selection minimizing Brier | 6/7 selected GBM + 1/7 selected RF; 25 ensemble-selection steps allowed |

LR's φ fits median imputation, missingness indicators, scaling and one-hot encoding
on training only. AutoGluon uses its pinned 1.6.1 feature generator, explicitly prefitted on
training inputs before passing it into the learner. Its default joint train/validation
feature fitting was corrected after a source audit; all 14 validation predictions
remained exactly identical. The original attempt is archived and the correction
is disclosed in `review/preprocessor_correction.json`. RF's versioned
adapter fills numerical nulls with zero; LightGBM uses native missing handling.
The published input table itself preserves nulls, and the LLM sees null explicitly.
These preprocessing differences are disclosed, not retrospectively harmonized.

AutoGluon runs four RF and four LightGBM candidates, plus its weighted ensemble.
The full comparison also includes four LR candidates and the prior. The search
has a 3600-second ceiling, eight CPU threads, no GPU, no outer bagging, stacking,
dynamic stacking, full refitting or calibration. Forest-internal bootstrap is
retained. All fitted defaults and fitted iterations appear in
`metadata/hyperparameters_all.json` and `metadata/model_info.json`.

Global $B^\star=\arg\min_{m\in\mathcal M}\widehat{BS}_{val,m}$ is the ensemble.
The validation leaderboard reports every candidate, not only winners. A tie would
use measured validation latency then name. No fit/selection uses test labels.

## Pretrained systems and exact controls

All four methods use the same Qwen3.8-27B weights, public revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; all 18 local shard hashes match the public
LFS hashes. The recovery provider is a task-owned Transformers 5.22.1 endpoint
with PyTorch 2.6.0+cu124 on four RTX 4090 GPUs, using BF16 and balanced layer
distribution. The original SGLang/RTX 5090 attempt is archived. Requested seeds
are grouped into native HF batches; batch position can affect sampled outputs.
The complete runtime and deviation records accompany the frozen protocol.
Sampling: temperature 0.7, top-p 0.95, top-k 20, thinking disabled, output cap 512
tokens/call, context 8192. Seeds are 17/42/2026. The training cutoff is unknown;
requested seeds do not ensure bitwise GPU determinism. Prompts and actual requests
are retained. The interface is a JSON action loop over HTTP chat completions.

| ID | Mathematical definition | Extra information / computation |
|---|---|---|
| D0 | $\hat p_i=\mathrm{parse}(L_\theta(P_0,X_i,\hat\pi;\xi_r))$ | One call from the input packet and prior |
| A1 | $a_k\sim\pi_\theta(\cdot\mid s_k),\ o_k=\mathcal T(a_k;T_i),\ s_{k+1}=U(s_k,a_k,o_k),\ \hat p_i=g_\theta(s_\tau)$ | At most four tool results and five planner calls |
| A2 | $z_i=V_\theta(X_i,H_i,\hat p_{i,A1});\ \hat p_{i,A2}=R_\theta(X_i,H_i,z_i)$ if a located error requires revision, otherwise  $\hat p_{i,A1}$ | Exact shared A1 prefix, separate-context verification, at most one revision |
| D_match | $\hat p_i=\mathrm{parse}(L_\theta(P_{match},X_i,\mathrm{canonical}(H_i);\xi_r))$ | One new call with actual A1 tool payloads, excluding planner draft/analysis |

No LLM training objective is optimized here; these are frozen pretrained inference
procedures. A2 is not a separately trained verifier, and its errors need not be
independent. Tool use and feedback are related to ReAct and Self-Refine; this code
does not claim to reproduce those papers' experiments or hidden reasoning prompts.

Tools: `get_proposal`, `get_actor_history`, `predict_tabular`, bounded `calculate`,
and `validate_evidence`. The evidence server owns identity/cutoff, allows no arbitrary
files/network/code and exposes the identical $B^\star$ used as a standalone baseline.
The tool field `training_only_fit` refers to base-model fitting and input
preprocessing. Candidate choice, stopping iterations and ensemble weights use
validation data as explicitly specified above; the flag does not mean validation
labels were never used for selection. One retry is allowed per failed transport call. Invalid JSON/probability/citation,
tool-budget exhaustion, prohibited arguments and timeout produce the same fixed
training-prior fallback; failed cases remain in every denominator.

```
for case, requested_seed:
    D0 <- one prediction from X and training prior
    A1 <- plan; obtain <=4 server-bound tool results; stop with probability
    A2 <- verify the exact A1 draft in a separate context
          revise once only if a located evidence error requires it
    D_match <- one prediction from X and A1's actual canonical tool results
    validate each output; otherwise use training prior
    save every physical HTTP request/response, payload, output and failure
```

## Scores, uncertainty and cost

For $R_m$ repeats, first average *loss*,

$$
L_{im}=R_m^{-1}\sum_r(p_{imr}-Y_i)^2,\quad
\widehat{BS}_m=N^{-1}\sum_iL_{im},\quad
\Delta_m=\widehat{BS}_m-\widehat{BS}_{B^\star}.
$$

Negative Δ favors the method. Averaging probabilities first would evaluate a
different multi-run ensemble and is not the primary estimand. Secondary metrics:
log loss with epsilon $10^{-6}$, AUROC, average precision, balanced accuracy at
0.5, calibration and run-level variation. Check `configs/statistics.json` for the
actual epsilon if this document is compared against another run.

Let $g=1,\dots,G$ index questions, $n_g$ their sizes, $S_{gm}=\sum_{i\in g}L_{im}$.
Draw G questions with replacement 9999 times using the same draws for all methods:

$$
\widehat{BS}^{(b)}_m=\frac{\sum_{j=1}^G S_{g_j^{(b)},m}}
{\sum_{j=1}^G n_{g_j^{(b)}}}.
$$

The denominator varies with each draw. Percentile 2.5/97.5% limits form marginal
conditional intervals. For a paired contrast $d_i=L_{ia}-L_{ib}$, δ is its mean,

$$
\widehat{SE}^2=\frac{G}{G-1}\frac{\sum_g(\sum_{i\in g}d_i-n_g\delta)^2}{N^2},
\quad t_b^*=\frac{\delta_b^*-\delta}{SE_b^*},\quad
p=\frac{1+\sum_b\mathbb1\{|t_b^*|\ge|\delta/SE|\}}{B+1}.
$$

Resampled copies are distinct groups for $SE_b^*$. Fewer than 30 groups, zero
observed SE or any degenerate bootstrap SE suppresses p-values. The main contrast
is A2−B*. The secondary family {A1−D0,A2−A1,A1−D_match} uses Holm correction
only when all three raw p-values are available; if one is suppressed, the entire
family's adjusted tests are withheld. In the completed run, one degenerate A2−A1
draw triggered this rule. Its percentile interval remains descriptive; the two
other raw p-values are not substituted for family-adjusted tests. Intervals are
not simultaneous. Seven/14/28-day sensitivity merges calendar bins
traversed by complete questions, never splitting a question. Group weighting is
an alternative descriptive estimand. Few calendar blocks cannot support strong
significance claims. Intervals condition on fitted models and observed runs, not
all sources of pretraining/fitting uncertainty.

History-availability diagnostics use `proposer_prior_completed == 0`; requester
history is a distinct input and is not silently combined into that indicator.
Period diagnostics split chronologically ordered test requests into first/second
halves. Calibration bins pool individual run outputs, report unique requests
separately, and do not treat repeated outputs as independent requests.

Confidence intervals describe estimated performance/contrasts. We do not report
prediction intervals for a binary probability output or fabricate coverage/width.

A2 cost includes A1+verifier+revision. D_match system cost includes obtaining A1
evidence plus its new call; its marginal-call latency is also recorded. Physical
calls are counted only once in the experiment ledger. Timing includes retries,
tools and verification, excludes server/model cold start, and is measured under
the stated concurrency/hardware. Unknown cached tokens stay null. No dollar tariff
was supplied for the owned GPUs, so dollars remain unavailable; allocated GPU-hours
are walltime times four, not measured busy utilization or rented-cost estimates.
Known token sums include the successful calls inside an output with incomplete
usage reporting. A total remains null if any call's usage is unknown. Token means
use only outputs with complete provider usage, with their denominator reported;
this restriction never removes outputs from score or latency calculations.

Original references and stable identifiers are in `paper/references.bib`; exact
implementation and execution identity are in `metadata/frozen_protocol.json`.
