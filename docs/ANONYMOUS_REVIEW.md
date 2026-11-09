# Review distribution and anonymity

The working repository and the anonymous review artifact serve different purposes.
A personal GitHub account, its profile and its commit history are identifying;
an anonymous author block in the PDF does not conceal the hosting account.
Private access also does not satisfy unauthenticated reviewer access.

Create a separate content-prepared review ZIP from a committed checkout:

```bash
python3 scripts/export_review.py --output /absolute/path/OraclePredictBench_Review.zip
```

The export excludes Git history, GitHub workflow metadata, fitted model binaries
(which contain original local paths), and the original export inventory. Private
correspondence, author registration lists, secrets and remote-operation scripts
are already absent from the working snapshot. Textual home-directory paths and
working repository URLs are redacted with before/after SHA-256 entries. Public
dataset identifiers, third-party references and copyright notices remain truthful.
Original freeze records are preserved as historical evidence, not rewritten to
pretend the redacted bytes were the inputs used for inference.

The ZIP retains complete saved prediction/call records, processed inputs, analysis
code, source data, training and inference code, and figure sources. `make setup`
and `make reproduce-results` work without fitted models. CPU refitting and new
LLM inference use the separate-workspace procedure in `REPRODUCE.md`; checking
omitted historical model binaries is unavailable in the review ZIP.

Content preparation alone cannot guarantee anonymity: task/dataset provenance and
published citations can still suggest authorship. The final hosting account,
visible history, download metadata and outbound links must also be checked.
A review host or dedicated anonymous account must be supplied and accessible
before an actual anonymous link can be inserted into the manuscript. Do not insert
a fabricated URL or describe a private personal repository as reviewer-accessible.
