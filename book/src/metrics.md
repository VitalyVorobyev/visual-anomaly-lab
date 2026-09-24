# Interpreting metrics

Metrics answer different questions. A single headline number is not a substitute for reading their
assumptions.

## ROC-AUC

ROC-AUC depends only on ranking and is insensitive to class prevalence. It can look excellent when defects
are rare even if the useful high-specificity operating region is weak. Report the subset and sample count,
and inspect ROC curves near the false-positive rate the application can tolerate.

## PR-AUC

Precision-recall changes with defect prevalence and therefore better reflects rare-positive retrieval. It
cannot be compared across datasets with different prevalence without saying so. Use it with expected
production prevalence or an explicitly fixed benchmark protocol.

## Pixel ROC-AUC

Pixel ROC-AUC treats pixels as ranking units. Large normal backgrounds can dominate; a method may score well
while producing imprecise boundaries. It is meaningful only after source-frame inverse projection and mask
alignment have been verified.

## AU-PRO

Per-region overlap gives each connected defect region a voice and integrates overlap under a bounded false
positive rate. It is often the more useful localization metric for industrial defects of different sizes.
It still depends on annotation conventions for holes, disconnected regions, and very small defects.

## Threshold metrics

Precision, recall, F1, specificity, and confusion matrices describe one operating point. Always report:

- the rule that selected the threshold;
- the subset on which it was selected;
- the resolved numeric value for that run;
- whether evaluation reused the selection subset.

Choosing a threshold on test labels and reporting it as deployment performance is optimistic. The workbench
makes threshold provenance visible but cannot turn an invalid protocol into a valid one.

## Uncertainty and seeds

One run measures one seed and one split. For stochastic training, report median and full range at small `n`
rather than mean ± standard deviation that implies more certainty than exists. Predeclare a promotion floor
or comparison rule before reading candidate results. A small delta inside seed spread is a hypothesis, not a
win.

## Resource evidence

Quality is not the only outcome. Record prepared size, batch, device/provider, train and inference time,
peak resident and accelerator memory, checkpoint and artifact sizes, asset identity, and version pins. A
method that swaps or cannot deploy is not a useful reference on that machine.

The generated [Public benchmarks](generated/benchmarks.md) page distinguishes protocol-quality results from
exploratory measurements and links each row to its full evidence log.
