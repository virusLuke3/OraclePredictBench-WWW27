# OraclePredictBench — WWW short-paper reproduction

Code and execution evidence for **Evaluating Predictive Models and AI Agents for Blockchain Oracle Adjudication**, matching the November 9 manuscript. This snapshot includes the November 6 predictor-access ablation and Verifier analysis, and the final green/blue method figure.

The target is eventual **UMA protocol disagreement**, conditional on a retrospectively selected challenged/settled cohort. It is not factual truth, and the reused historical test makes these exploratory comparisons.

## Start here: reproduce saved results on CPU

Use Linux, Python 3.12 and `uv`. Run from the repository root:

```bash
make setup
make reproduce-results
```

No GPU, API key or running model is needed. This installs a separate `.venv`, checks every saved test prediction against its raw case record, recomputes both batches' statistics and paired intervals, and redraws both figures. It fails if the 15 checked result tables, two generated LaTeX files, bootstrap draws or figure pixels differ from the supplied reference. The receipt is `validation/replay_report.json`.

This **recalculates results from saved predictions**. It does not rerun LLM inference, remeasure latency or recreate an untouched test.

To compile the four-page manuscript with an installed Tectonic engine:

```bash
make paper TECTONIC=/absolute/path/to/tectonic
```

Read [the complete reproduction guide](docs/REPRODUCE.md) for baseline refitting, fresh LLM inference, dependencies, resource requirements and anonymous review packaging. A new inference run must use a separate workspace.

## What is included

| Component | Evidence / implementation |
|---|---|
| Current paper and two figures | [`paper/main.pdf`](paper/main.pdf), [`paper/main.tex`](paper/main.tex), [`paper/figures/`](paper/figures/) |
| Fixed public release and checksums | [`metadata/public_input_reference.json`](metadata/public_input_reference.json), [`metadata/data_manifest.json`](metadata/data_manifest.json) |
| Feature reconstruction, label definition and split audit | [`src/oraclepredict/data.py`](src/oraclepredict/data.py), [`metadata/feature_availability_audit.csv`](metadata/feature_availability_audit.csv), [`metadata/split_manifest.csv`](metadata/split_manifest.csv) |
| Prior, logistic regression, RF, LightGBM, weighted ensemble | [`src/oraclepredict/baselines.py`](src/oraclepredict/baselines.py), [`configs/baseline.json`](configs/baseline.json) |
| Direct LLM, planner, verifier, matched-evidence control | [`src/oraclepredict/agents.py`](src/oraclepredict/agents.py), [`src/oraclepredict/tools.py`](src/oraclepredict/tools.py), [`prompts/`](prompts/) |
| Predictor-access removal, fresh interleaved controls | [`src/oraclepredict/revision_ablation.py`](src/oraclepredict/revision_ablation.py), [`configs/revision_20261106/ablation.json`](configs/revision_20261106/ablation.json) |
| Original 471 case/repeat records, 1,884 LLM outputs | [`runs/agents_test/`](runs/agents_test/), [`runs/agents_test.parquet`](runs/agents_test.parquet) |
| Fresh 942 case/repeat/arm records, 1,884 outputs | [`runs/revision_20261106/agents_test/`](runs/revision_20261106/agents_test/), [`runs/revision_20261106/agents_test.parquet`](runs/revision_20261106/agents_test.parquet) |
| Verifier changes, fallbacks, physical/tool calls | [`Results/revision_20261106/`](Results/revision_20261106/), [`scripts/analyze_revision.py`](scripts/analyze_revision.py) |
| Scoring and statistical formulas | [`docs/methods_math.md`](docs/methods_math.md), [`src/oraclepredict/evaluate.py`](src/oraclepredict/evaluate.py) |
| Figure sources | [`scripts/draw_method_schematic.py`](scripts/draw_method_schematic.py), [`scripts/plot_revision.py`](scripts/plot_revision.py) |
| Historical outage records, excluded from current results | [`runs/attempt01_outage/`](runs/attempt01_outage/), [`review/operational_attempts.json`](review/operational_attempts.json) |

Each case JSON contains actual requests, responses, tool payloads, retries and output records. Duplicate per-call copies have been omitted; the complete calls remain embedded in each case. Original freeze records and SHA-256 references are retained. Author correspondence, credentials and private remote-operation helpers are excluded.

## Results represented by this snapshot

The retained train/validation/test partitions contain **474/153/157 requests**; test has **120 question groups**. Each LLM method has three seeded repeats. Brier score is lower-is-better.

| Comparison | Mean Brier difference | Paired 95% interval |
|---|---:|---:|
| Original A2 − selected conventional predictor | +0.000456 | [−0.000802, +0.001677] |
| Fresh A1: predictor removed − available | +0.084602 | [+0.047940, +0.122536] |
| Fresh A2: predictor removed − available | +0.083077 | [+0.047330, +0.119727] |
| Fresh verifier A2 − A1, with predictor | +0.000328 | [−0.000084, +0.000922] |
| Fresh verifier A2 − A1, without predictor | −0.001197 | [−0.007217, +0.004057] |

Removing predictor access also changes the planner's available choices; it is an access-and-policy intervention. The results do not establish an orchestration advantage beyond the predictor. Verifier intervals cross zero; this does not establish equivalence. The new batch retains all five fallback outputs. Timing is recorded warm-run timing on different CPU/GPU hardware, not a matched-hardware throughput comparison.

## Provenance and access

The public data is pinned to `atlas-utc-20261029`, commit `a8e4e93016b7adfcb651bff9bafd26c2fab06093`. Third-party data and model citations are retained. See [NOTICE](NOTICE) and [LICENSE](LICENSE).

Repository availability and review anonymity are separate. A repository under a personal GitHub account is **not anonymous**, even if the manuscript has anonymous authors. The review export removes Git history, personal-path metadata and fitted binary models; see [review packaging](docs/ANONYMOUS_REVIEW.md). No unauthenticated anonymous reviewer access is claimed until a separate review URL has actually been hosted and checked.
