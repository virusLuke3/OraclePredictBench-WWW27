Overleaf source package — green/blue workflow figure, 9 November 2026

1. Create a new Overleaf project with Upload Project and choose this ZIP.
2. Set the main document to main.tex and the compiler to XeLaTeX.
3. Recompile. Bibliography is generated from references.bib with BibTeX.

The anonymous PDF has four pages including all references. The source uses the
standard ACM sigconf anonymous review layout. Unmodified acmart 1.83 and its
matching source (vendor/acmart.dtx) are included to preserve the tested layout;
ACM-Reference-Format.bst is also included. No class, margin or base-font changes
were made to compress the paper. Figures are original vector PDFs, with editable
SVG counterparts. They were generated from the supplied saved experiment data.

The local verification uses Tectonic 0.17.0 (XeTeX-based), default bundle v33:
  tectonic -k main.tex
The archive was extracted and recompiled locally. Online Overleaf compilation or
synchronization is not claimed. No model execution is needed to compile the paper.
The FigureSources_20261109_green_blue ZIP contains both plotting scripts, content
decisions and a claim-to-evidence record, seven saved CSV inputs, the plotted-value
record, editable SVGs and vector PDFs. Figure 1 restores the user-selected green/blue
two-row layout, black line icons, rounded boxes and arrows. Its labels describe
executed workflow and checks: historical temporal splits, recorded outcomes,
paired intervals, warm-run latency, feature auditing, Brier scoring and agent
comparisons on a reused test. The caption states the retrospective scope and
actual controls. The figure contains no numbers or success symbols.
Figure 2 is unchanged from the preceding revision. Earlier archives preserve
previous layouts and experiments. No new experiments run in this figure update.

Files needed for compilation:
  main.tex
  generated_revision.tex
  tables/revision_exact.tex
  figures/method.pdf
  figures/results.pdf
  references.bib
  acmart.cls
  ACM-Reference-Format.bst

The manuscript concerns eventual UMA protocol disagreement in a retrospective
challenged/linked/settled/reconciled cohort. The reference outcome is not an
independently established external-truth label. All new comparisons remain
exploratory because the historical test cohort had been seen previously.

Upstream class source, distributed unmodified:
https://github.com/borisveytsman/acmart/releases/tag/v1.83
Template license and source notices remain in vendor/acmart.dtx and acmart.cls.
Proposed author information is intentionally kept outside this anonymous source.
