# Reproduction guide

## A. Recompute the supplied paper results (CPU; no model calls)

```bash
make setup
make reproduce-results
make paper TECTONIC=/absolute/path/to/tectonic
```

`make setup` creates a Python 3.12 environment from `requirements-replay.lock`.
`make reproduce-results` verifies 3,768 LLM output records against 1,413 raw case
records, recomputes 9,999 paired question-cluster bootstrap draws for each batch,
calendar sensitivity, verifier changes/fallbacks, physical-call accounting,
exact result tables and both publication figures. The complete call payloads are
inside the case JSONs. `validation/replay_report.json` records successful checks;
`validation/replay.log` records subprocess output.

The command deliberately preserves measured CPU latency and offline preparation
measurements. Recomputing a statistic is not remeasuring a system. Conditional
intervals do not include every source of model-training or sampling uncertainty.
The original primary approximate p-value is reported; a degenerate secondary
comparison prevents the complete prespecified Holm family from being reported.
The new access intervention reports exploratory intervals, not a new significance
family. The historical test had already been examined.

The bundled processed inputs and predictions make this route independent of the
original research server. Dependency installation may require network access;
the computation itself needs no model/network service. Tectonic 0.17.0 was the
reference compiler and may download its TeX bundle on first use. Native font and
renderer differences can change image pixels; the check fails rather than silently
claiming exact visual reproduction.

## B. Independently refit the conventional models (CPU)

```bash
make setup-full
PYTHONPATH=src .venv/bin/python scripts/refit_check.py
make test
make check-frozen
```

`setup-full` adds the original `requirements.lock`. Refitting creates a separate
workspace, trains all 14 candidates, checks all validation probabilities and all
785 baseline test probabilities against the stored reference, and verifies the
selected model. It never retrains inside the supplied `models/` directory. Numeric
tolerances are `rtol=1e-7, atol=1e-8`. Timing is not expected to match.

`make test` exercises predictor-access blocking, prompt isolation, train-only
preprocessing, temporal history availability, grouping, scoring and evidence-ID
validation. It requires AutoGluon and fitted models. In the anonymous export,
fitted binaries are omitted: run tests in the separate refit workspace printed
by `refit_check.py`, using this environment's Python and that workspace's `src`.
`check-frozen` requires all original fitted binaries and is available in the full
working repository; the original freeze hashes are retained in the review export
for provenance but cannot validate omitted binaries.

Raw public inputs are bundled in the full working repository. If absent in a
review export, `make data` downloads them at the pinned release commit. The download
checks published hashes/sizes and writes a new timestamped retrieval manifest.
Run raw feature reconstruction in a separate new workspace, because metadata and
processed outputs are regenerated:

```bash
.venv/bin/python scripts/prepare_new_run.py --output ../oracle-reconstruction
cd ../oracle-reconstruction
make setup
make data
PYTHONPATH=src .venv/bin/python -m oraclepredict.data
```

## C. Rerun models and LLM inference (separate workspace; GPU required)

This is a new stochastic experiment, not route A. The code is supplied; a new full
LLM run was **not** performed during repository packaging. Preserve the reference
results and create an empty run workspace:

```bash
.venv/bin/python scripts/prepare_new_run.py --output ../oracle-new-run
cd ../oracle-new-run
make setup-full
make data
PYTHONPATH=src .venv/bin/python -m oraclepredict.data
PYTHONPATH=src .venv/bin/python -m oraclepredict.baselines train
```

The reference model is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The original run used BF16 on four
RTX 4090 GPUs, Transformers 5.22.1 and PyTorch 2.6.0+cu124. Its separately recorded
serving environment is `metadata/backup_runtime_requirements.txt`. The checkpoint
is roughly 56 GB and is not bundled. The serving environment is separate from the
CPU analysis environment; CUDA libraries/drivers must be compatible with the host.

Set `ORACLEPREDICT_MODEL_PATH` explicitly, retrieve the pinned checkpoint, verify
its weight hashes, then start the supplied localhost-only server. For example,
after creating a compatible serving environment named `.venv-model`:

```bash
export ORACLEPREDICT_MODEL_PATH=/absolute/path/to/Qwen3.8-27B
PYTHONPATH=src .venv/bin/python scripts/download_model_backup.py
PYTHONPATH=src .venv/bin/python scripts/verify_backup_snapshot.py
.venv-model/bin/python scripts/serve_transformers_backup.py
```

The download script may obtain weight bytes through the Qwen transport mirror;
identity comes from pinned SHA-256 values, not its floating branch name. Verify
all weights before serving. Record the **actual new** runtime/hardware versions
and readiness in `metadata/backup_runtime_identity.json`; the old record was moved
to `reference_metadata/` by the workspace helper and must not be passed off as a
measurement of the new machine. Use a separate terminal for the commands below:

```bash
export ORACLEPREDICT_BASE_URL=http://127.0.0.1:31028/v1
export PYTHONPATH=src
.venv/bin/python -m oraclepredict.agents --split pilot --repeats 1
.venv/bin/python scripts/summarize_pilot.py
# Inspect actual pilot failures and runtime identity before freezing.
.venv/bin/python -m oraclepredict.freeze
.venv/bin/python -m oraclepredict.baselines predict --split test
.venv/bin/python -m oraclepredict.agents --split test --repeats 3
.venv/bin/python -m oraclepredict.revision_ablation pilot
.venv/bin/python -m oraclepredict.revision_ablation freeze
.venv/bin/python -m oraclepredict.revision_ablation test
.venv/bin/python scripts/measure_revision_costs.py
.venv/bin/python -m oraclepredict.evaluate
.venv/bin/python scripts/analyze_revision.py
.venv/bin/python scripts/plot_revision.py
.venv/bin/python scripts/write_revision_results.py
```

The original main batch produced 1,884 outputs; the fresh two-arm intervention
produced another 1,884. Model batching can change seeded sampling; regenerated
probabilities and timings need not be identical. Never run the reference-exact
`reproduce_paper.py` checker against new stochastic outputs and claim they should
match. Invalid outputs, retries and prior fallbacks remain in denominators.
