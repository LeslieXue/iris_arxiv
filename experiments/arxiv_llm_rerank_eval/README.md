# LLM-Guided Explanatory Chunk Reranking for IRIS

## Experiment status

**Status: archived after the feasibility pilot (2026-10-19).**

The code and prompts are retained as an experimental record. The LLM signal has
not been added to the IRIS training loss because the current experiment can show
that Qwen changes IRIS selections, but it cannot independently establish that
the changed selections are more explanatory.

## Research question

IRIS learns query vectors that retrieve a small set of document chunks for
classification. The selected chunks can be traced back to their original text,
but the classification objective alone does not guarantee that they provide the
kind of explanation a human would prefer.

This experiment asked whether a local LLM teacher could add a useful inductive
bias. Given the known document category and several chunks retrieved by IRIS,
the teacher should prefer chunks that explain *why the paper belongs to that
category* through concrete technical content.

The intended evidence should describe some combination of:

- the paper's research object or problem;
- its method, model, algorithm, or experimental design;
- its result or technical contribution; and
- the connection between that content and the known target category.

The teacher should avoid rewarding a chunk merely because it contains the
category name, obvious keywords, section metadata, or other superficial cues.

The long-term proposal was to convert the teacher scores into offline ranking or
preference supervision and optimize IRIS with a combined objective such as:

```text
classification loss + lambda * teacher preference loss
```

Qwen would be used only to prepare supervision. The final IRIS model would not
require Qwen at inference time.

## Experimental setup

### Data and trained IRIS model

- Dataset: arXiv11 subset with `cs.AI`, `cs.DS`, and `cs.PL`
- Documents: 10,032
- Vector database rows: 857,189 chunks
- Encoder: `answerdotai/ModernBERT-base`
- Embedding dimension: 768
- IRIS learned queries: 8
- Baseline test accuracy: 0.9024
- Baseline balanced accuracy: 0.8977
- Baseline macro-F1: 0.8967

Each stored vector row retains `document_id`, `chunk_id`, `source_path`, and
`chunk_text`, allowing every retrieved vector to be traced directly to its
source text.

### Teacher model

- Model: `Qwen/Qwen3-8B`
- Execution: local model inference on a DCC RTX 5000 Ada GPU
- Precision: bfloat16
- Output format: structured JSON

The target-conditioned teacher returns:

- `target_evidence` (0-4)
- `substantive_relevance`
- `specificity`
- `context_completeness`
- `direct_label_disclosure`
- `keyword_only_risk`
- `brief_reason`

## Development history

### 1. Candidate export and reproducible audit data

IRIS candidates were exported once and stored with fixed document, query, chunk,
rank, and text identifiers. This separated retrieval from teacher scoring, made
prompt versions comparable, and prevented candidate drift between runs.

The initial export covered 240 documents:

- validation: 30 documents per class;
- test: 50 documents per class; and
- candidate counts per document: minimum 11, median 44, maximum 64.

Generated data and model outputs are stored under `/work` on DCC and are not
committed to Git.

### 2. Abandoned multiclass-scoring formulation

The first formulation asked Qwen to score evidence for `cs.AI`, `cs.DS`, and
`cs.PL`. Prompt versions `teacher_v1`, `teacher_v2`, and `teacher_v3` were
used to study this behavior.

This formulation was abandoned because it treated each chunk as a standalone
classification example. That is not the intended role of the teacher. The IRIS
candidates for a document all support the same document-level target; the
teacher should distinguish explanatory value within that target rather than
reclassify each chunk.

The multiclass audit also revealed a tendency to predict `cs.DS` for technical
chunks from `cs.AI` and `cs.PL`. Improving document-label alignment would not
necessarily improve the explanatory quality of the selected chunks, so those
metrics were retained only as prompt-development diagnostics.

### 3. Target-conditioned scoring

The experiment was redesigned so that Qwen receives the known target category
and assigns a single `target_evidence` score. The core instruction became:

> Given the paper's known category, identify chunks whose concrete technical
> content best explains why this paper belongs to that category, without relying
> on category names, keywords, or metadata alone.

`target_evidence_v1` exposed a calibration problem: Qwen frequently lowered a
score because a chunk did not explicitly mention the category name or familiar
category terminology.

`target_evidence_v2` clarified that:

- the target label is an instruction and is not itself evidence;
- explicit category words are unnecessary;
- a clear research object plus a concrete method, result, or contribution can
  receive a high score; and
- missing category terminology must not be cited as a weakness.

On the 36-chunk prompt-development audit, the score distribution changed as
follows:

| Prompt | Score 0 | Score 1 | Score 2 | Score 3 | Score 4 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| `target_evidence_v1` | 4 | 8 | 20 | 4 | 0 | 1.667 |
| `target_evidence_v2` | 3 | 0 | 13 | 20 | 0 | 2.389 |

Reasons referring to missing category cues decreased from 13 to 5. The average
scores also became similar across the three document categories. Manual review,
however, found a likely false negative involving specialized
programming-languages theory, demonstrating that the teacher can undervalue
technically valid evidence outside its most recognizable patterns.

The prompt was frozen at `target_evidence_v2` rather than repeatedly tuning it
to individual audit examples.

### 4. One-document top-32 reranking pilot

A held-out validation paper not used for prompt development was selected:

- document ID: 1324
- target: `cs.AI`
- source: `cs.AI/train_012948.txt`

The `train_` filename records the source dataset's original split; this document
belongs to the fixed IRIS validation split used by this experiment.

For each of the 8 learned IRIS queries, the top 4 chunks were collected, giving
32 unique candidates. The comparison preserved the query structure:

- original selection: IRIS rank 1 for each query;
- Qwen selection: highest `target_evidence` among that query's four candidates;
- tie rule: retain the candidate with the better original IRIS rank.

This produces two eight-chunk explanation sets without changing the number of
IRIS queries or the explanation budget.

## Pilot results

### Score distribution

Across the 32 candidates:

| Target-evidence score | Count |
|---:|---:|
| 0 | 4 |
| 1 | 0 |
| 2 | 14 |
| 3 | 14 |
| 4 | 0 |

Seven of the eight query groups contained more than one score level, showing
that Qwen produced a non-constant preference signal.

### Selection changes

- changed queries: 4 of 8
- unchanged queries: 4 of 8
- unique Qwen maximum: 2 of 8 query groups
- tied Qwen maximum: 6 of 8 query groups
- all four changes moved from original IRIS rank 1 to rank 2
- every changed selection moved from teacher score 2 to teacher score 3
- mean teacher score of original selections: 2.25
- mean teacher score of Qwen selections: 2.75

The increase from 2.25 to 2.75 only confirms that the reranker follows its own
score. It is not independent evidence that the new selections are better.

### Qualitative observations

The four changed query groups showed different kinds of behavior:

1. In one clear case, Qwen replaced a truncated mathematical passage with a
   coherent description of classifier training, robot manipulation strategy,
   simulation, and experimental findings. This matched the intended preference
   for a complete technical explanation.
2. In another case, Qwen replaced a concrete self-supervised optimization method
   with a related-work passage containing recognizable terms such as intuitive
   physics, mental simulation, and transfer learning. The replacement looked
   more obviously related to AI, but it was less focused on the paper's own
   contribution.
3. One change replaced the paper's label-ranking computation with a weighted
   Kendall correlation measure. The second passage was coherent, but it described
   evaluation rather than the central method.
4. One change replaced concrete learned robot manipulation preferences with a
   passage about scene clustering, search-tree growth, and deployment efficiency.
   Both passages were technically meaningful but explained different aspects of
   the work.

These cases show that Qwen can move selections toward more complete technical
text, but can also favor recognizable terminology, related work, or auxiliary
evaluation methods over the paper's central contribution.

## What the experiment established

The feasibility pilot established that:

- original IRIS chunks can be scored locally by Qwen with reliable structured
  output;
- Qwen scores can be joined back to exact `document_id` and `chunk_id` records;
- the teacher produces non-constant preferences within IRIS query groups;
- a reranker can change the selected explanation while preserving the IRIS query
  structure and chunk budget; and
- some changed selections qualitatively follow the intended explanatory bias.

## Main obstacle

The central unresolved issue is evaluation.

There is no gold chunk-level annotation indicating which passage best explains
why an arXiv paper belongs to its document category. The document label is not a
gold label for every individual chunk. Teacher-score improvements cannot serve
as independent validation because the same Qwen model both defines the score and
selects the winner.

The current alternatives all have limitations:

- classification accuracy tests predictive utility, not explanatory quality;
- another LLM judge may reproduce similar model and prompt biases;
- automatic keyword or label-alignment metrics reward the shortcuts the project
  is trying to avoid;
- self-evaluation by Qwen is circular; and
- a single manually inspected paper is too small to establish generalization.

The 0-4 scale is also coarse. Six of eight pilot query groups had tied maximum
scores, so it creates preference tiers rather than a stable total ordering.

## Why the experiment is archived here

Connecting these scores to the IRIS loss now would answer whether this particular
teacher signal changes training, but a positive training result would still not
show that the selected chunks are more explanatory. A negative result would also
be ambiguous: it could reflect the teacher, prompt, score resolution, loss
construction, or optimization weight.

For that reason, the project stops after the feasibility and descriptive pilot.
No teacher loss is added to IRIS in this version.

## Requirements for a stronger follow-up

A future study should first establish an independent evaluation protocol. A
minimal design would include:

1. multiple held-out papers from every class;
2. blinded pairs that hide whether a chunk came from IRIS or Qwen reranking;
3. a fixed rubric covering the paper's research object, core method, main result,
   contribution, context completeness, and shortcut use;
4. at least two independent human raters or one domain expert plus a documented
   adjudication procedure;
5. inter-rater agreement and confidence intervals; and
6. frozen prompts and a held-out test set that is not used for prompt revision.

Only after the teacher preference agrees sufficiently with independent judgments
should the project proceed to a pairwise/listwise teacher loss and compare:

- baseline IRIS;
- IRIS with shuffled or random teacher scores;
- IRIS with Qwen preference supervision; and
- different teacher-loss weights.

Predictive performance and explanatory quality should be reported separately.

## Repository map

### Target-conditioned experiment

- `config.target.toml`: paths and target-conditioned experiment settings
- `prompts/target_evidence_v1.txt`: initial target-conditioned prompt
- `prompts/target_evidence_v2.txt`: frozen calibrated prompt
- `build_target_audit.py`: build the prompt-development audit set
- `score_target_candidates.py`: run local Qwen scoring
- `analyze_target_audit.py`: summarize score distributions and diagnostic flags
- `build_top32_pilot.py`: build the held-out 8-query x 4-candidate pilot
- `describe_top32_pilot.py`: compare IRIS and Qwen selections descriptively

### Earlier multiclass prompt study

- `prompts/teacher_v1.txt`
- `prompts/teacher_v2.txt`
- `prompts/teacher_v3.txt`
- `build_audit_set.py`
- `score_candidates.py`
- `analyze_audit.py`
- `compare_prompt_versions.py`

### Slurm jobs

Corresponding DCC job definitions are under:

```text
slurm/arxiv_llm_rerank_eval/
```

## Artifact policy

Candidate text, Parquet files, Qwen outputs, metadata, reports, model caches, and
checkpoints remain in the DCC `/work` directory and are excluded from Git. The
repository retains the code, prompts, configuration, and this summary so the
experimental decisions and limitations remain reproducible and reviewable.
