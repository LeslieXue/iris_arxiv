#!/bin/bash

set -eo pipefail

cd /hpc/home/jx198/iris_arxiv

EXPORT_JOB=$(sbatch --parsable \
    slurm/arxiv_llm_teacher_loss_one_third/export_one_third_chunks.sbatch)
SCORE_JOB=$(sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    slurm/arxiv_llm_teacher_loss_one_third/score_one_third_array.sbatch)
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${SCORE_JOB}" \
    slurm/arxiv_llm_teacher_loss_one_third/merge_scores.sbatch)
TRAIN_JOB=$(sbatch --parsable \
    --dependency="afterok:${MERGE_JOB}" \
    slurm/arxiv_llm_teacher_loss_one_third/train_variants.sbatch)
EVAL_JOB=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    slurm/arxiv_llm_teacher_loss_one_third/evaluate_variants.sbatch)
SUMMARY_JOB=$(sbatch --parsable \
    --dependency="afterok:${EVAL_JOB}" \
    slurm/arxiv_llm_teacher_loss_one_third/summarize_results.sbatch)

echo "Export/reuse job: ${EXPORT_JOB}"
echo "Qwen score array (7 x RTX 5000 Ada, batch 12): ${SCORE_JOB}"
echo "Merge job: ${MERGE_JOB}"
echo "Training array (3 variants): ${TRAIN_JOB}"
echo "Evaluation array: ${EVAL_JOB}"
echo "Summary job: ${SUMMARY_JOB}"
