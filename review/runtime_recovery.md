# Complete-run recovery and implementation correction

The first test attempt is preserved in `runs/attempt01_outage`, including all
471 case–repeat records, 2,839 physical requests, 1,884 method outputs and the
fixed fallback predictions. Connection failures affected 2,508 requests and
1,671 method outputs. A full rerun was chosen before scoring that attempt; the
decision timestamp is in `operational_attempts.json`. The first attempt was
subsequently scored to make its failure-inclusive results auditable. No
successful prefix is substituted for its intended denominator.

A later source audit found that AutoGluon's default unsupervised feature
generator fitted combined train/validation inputs. Test inputs were not used.
The corrected implementation prefits the generator on the 474 training requests
only, then passes the fitted generator to AutoGluon. The original candidate grid
was refitted. All 2,142 validation predictions across 14 candidates are exactly
unchanged, as is validation-selected B*. The correction was motivated by the
implementation audit, not by test-score selection. Its evidence is in
`preprocessor_correction.json`, `preprocessor_correction_comparison.csv`, the
training fit manifest and the new-environment reproduction report.

| Item | Original attempt | Recovery protocol |
|---|---|---|
| Data / splits / 11 allowed inputs | Pinned public release | Unchanged |
| Labels / score / paired cluster analysis | Frozen specification | Unchanged |
| Conventional search / validation choice | 14 candidates | Same search; strict train-only preprocessing correction |
| Qwen weights | Pinned 27B snapshot | Identical shard hashes |
| Prompts / tools / sampling / output limits | Versioned files | Unchanged |
| Serving engine | SGLang 0.5.13.post1 | Transformers 5.22.1 with task-owned JSON endpoint |
| PyTorch / GPUs | 2.11.0+cu130; four RTX 5090 | 2.6.0+cu124; four RTX 4090 |
| Model distribution | Tensor parallel 4 | Layer placement across four GPUs; no quantization |
| Sampling execution | SGLang request seeds | Native HF batch sampling, grouped by requested seed |
| Job submission | Case major | Repeat major to keep equal-seed batches together |
| Test coverage | All 157 × 3; outage fallback retained | All 157 × 3 required again, with a new validation pilot and freeze |

HF sampling is reproducible only to the extent supported by the recorded runtime
and scheduling: batch position can change draws even with the same requested
seed. Changing serving backends is not expected to reproduce the original token
stream. The backup applies a task-local import alias for the older JSON-format
enforcer and makes delta-rule inputs contiguous without changing their values.
Accelerated kernels were compared with the pure Torch reference at batch sizes
one and eight and the model's 48 value heads; numerical tolerances and measured
errors are retained. No shared Python installation was patched.

The new 32-request validation pilot completed 193 physical calls and 128 method
outputs in 1,052.5 seconds, with no transport errors or fallback outputs. Its
observed timing and budget estimate superseded the original server's estimate
before the new freeze. Actual full-run timing is recorded separately in
`execution_validation.json`; a pilot estimate is not the final runtime. Dollar
tariffs and GPU busy utilization are unavailable; allocation wall time and system
latency are reported separately. A timed-out client attempt can still consume
server compute. Such calls remain in the physical ledger with unknown token usage,
and successful retries do not erase their latency or failure record.

The corrected test is **retrospective and exploratory**. The earlier attempt's
outcomes have been seen, and this local freeze is not public preregistration or
an untouched confirmatory test. Every operational deviation and the original
failure-inclusive results remain available for review.
