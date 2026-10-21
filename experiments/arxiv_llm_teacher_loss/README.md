# Full-data Qwen teacher-loss experiment

This experiment tests whether a fixed Qwen preference can bias IRIS toward
retrieving chunks that more clearly explain a document's known arXiv category.
It is the training counterpart to the archived reranking feasibility study.

## Experimental question

For each training document, IRIS dynamically retrieves four chunks for each of
its eight learned queries. Qwen assigns every database chunk a fixed teacher
utility before IRIS training. During training, IRIS receives an extra ranking
loss that prefers a higher-scored chunk over a lower-scored chunk when both
were retrieved by the same query for the same document.

Qwen is never called inside an IRIS training step. The test set does not load
teacher scores, and the true test label is not an input to prediction.

## Why every chunk is scored

IRIS updates its query vectors after every optimizer step. New chunks can enter
the retrieved top-k as the queries move. Scoring only the initial top-k would
make the teacher signal disappear or become undefined later in training. The
pipeline therefore scores all chunks in the Lance vector database and verifies
exact key coverage before training begins.

Each score is keyed by `(document_id, chunk_id)`. The Qwen prompt receives the
document's training label as the target category and evaluates how well the
chunk provides substantive evidence for that target.

## Teacher utility

`target_evidence` is the primary ordinal score. Explanatory quality and
shortcut-risk fields only break ties:

```text
quality = mean(substantive_relevance, specificity, context_completeness)
risk = mean(direct_label_disclosure, keyword_only_risk)
teacher_utility = target_evidence + 0.1 * (quality - risk)
```

The tie-breaker weight is constrained to less than `0.125`, so the secondary
fields cannot reverse a one-point difference in `target_evidence`.

## Training objective

For query `q`, retrieved vectors `v_i`, and teacher utilities `u_i`, the
retrieval score is the negative mean squared distance:

```text
s(q, v_i) = -mean((q - v_i)^2)
```

For every same-document, same-query pair where `u_i > u_j`, the teacher term is:

```text
softplus(-(s(q, v_i) - s(q, v_j)) / temperature)
```

Pairs with tied or nearly tied utilities are ignored. The full objective is:

```text
classification loss
+ existing IRIS query-similarity penalty
+ teacher_weight * pairwise teacher-ranking loss
```

The discrete Lance retrieval is rerun with the updated queries throughout
training. The pairwise loss changes the learned queries, and later retrievals
therefore reflect the accumulated teacher preference.

## Full experiment matrix

| Variant | Teacher scores | Weight | Purpose |
| --- | --- | ---: | --- |
| `baseline` | disabled | 0.00 | same-code IRIS baseline |
| `teacher_w001` | real | 0.01 | weak teacher bias |
| `teacher_w005` | real | 0.05 | medium teacher bias |
| `teacher_w010` | real | 0.10 | stronger teacher bias |

All variants use the same seed, full 10,032-document dataset, train/validation/
test split, model architecture, and optimizer settings.

## Evaluation

Each variant produces two kinds of test-set output:

1. Independent task performance: accuracy, balanced accuracy, macro-F1,
   cross-entropy, confusion matrix, and per-class metrics.
2. Teacher-alignment diagnostics: mean teacher utility, mean target evidence,
   high-evidence rate, zero-evidence rate, and shortcut-risk fields among the
   chunks selected by the learned queries.

Mean teacher utility is first averaged across the chunks selected for each
test document and then averaged across documents. The same document-level
aggregation is reported overall and separately for `cs.AI`, `cs.DS`, and
`cs.PL`.

Teacher alignment only shows that the training objective changed retrieval in
the requested direction. Because Qwen supplies both the training preference
and this diagnostic, it is not independent proof that the explanations are
better.
The total training loss is not compared across variants because teacher-enabled
runs contain an extra nonnegative term. Logs report classification loss, the
original query penalty, raw teacher-ranking loss, and weighted teacher loss
separately. Cross-entropy and held-out task metrics remain comparable.

## DCC pipeline

The pipeline uses eight Qwen scoring shards, then trains and evaluates all four
IRIS variants as Slurm arrays. From the repository root on DCC:

```bash
bash slurm/arxiv_llm_teacher_loss/submit_full_experiment.sh
```

The command prints all six Slurm pipeline job IDs. Dependencies enforce this order:

```text
export every chunk
  -> score eight Qwen shards
  -> merge and validate complete score coverage
  -> train five variants
  -> evaluate five variants
  -> create the comparison table
```

Final comparison files are written to:

```text
/work/jx198/iris_arxiv/results/teacher_loss_v1/comparison.json
/work/jx198/iris_arxiv/results/teacher_loss_v1/comparison.csv
/work/jx198/iris_arxiv/results/teacher_loss_v1/comparison.md
/work/jx198/iris_arxiv/results/teacher_loss_v1/primary_tradeoff.png
/work/jx198/iris_arxiv/results/teacher_loss_v1/primary_tradeoff.pdf
```
