# One-third IRIS teacher-loss experiment

This pipeline runs a faster intervention experiment while preserving a valid
held-out evaluation:

- recreate the original seed-42 80/10/10 document split;
- select one third of the training documents with label stratification;
- keep the complete validation and test splits unchanged;
- score every chunk belonging to the selected training documents;
- reuse valid Qwen scores from the cancelled full-data run;
- run Qwen3-8B on seven RTX 5000 Ada GPUs with batch size 12;
- compare matched baseline, shuffled-score, and real-teacher variants;
- evaluate classification on the complete test split;
- measure Qwen-preference alignment on the scored training subset only.

The alignment metric is a mechanism/compliance metric. It is not independent
evidence that explanation quality improved.

## Submit

```bash
cd /hpc/home/jx198/iris_arxiv
bash slurm/arxiv_llm_teacher_loss_one_third/submit_experiment.sh
```

The command prints six Slurm job IDs. Jobs run in this order:

```text
export subset and reuse old scores
  -> score missing chunks
  -> validate and merge scores
  -> train three variants
  -> evaluate three variants
  -> write comparison table
```

## Verify the export

After the export job completes, inspect:

```bash
cat /work/jx198/iris_arxiv/experiments/arxiv_llm_teacher_loss/v1_one_third/export_metadata.json
```

The important fields are `selected_train_documents`, `records`,
`reused_records`, `records_to_score`, and `selected_document_label_counts`.

## Monitor Qwen scoring

Replace `<score_job>` with the score-array job ID:

```bash
squeue -j <score_job> -o "%.20i %.24j %.10T %.12M %.30R"

for i in {0..6}; do
    file="/work/jx198/iris_arxiv/logs/one-third-score-<score_job>_${i}.out"
    echo "===== shard ${i} ====="
    if [[ -f "${file}" ]]; then
        grep "Batch " "${file}" | tail -1
    else
        echo "Not started"
    fi
done
```

Each score file is append-only and flushed after every successful record. A
failed or cancelled shard can be resubmitted safely; completed keys are skipped.

## Results

The final comparison is written to:

```text
/work/jx198/iris_arxiv/results/teacher_loss_v1_one_third/comparison.md
/work/jx198/iris_arxiv/results/teacher_loss_v1_one_third/comparison.csv
/work/jx198/iris_arxiv/results/teacher_loss_v1_one_third/comparison.json
```
