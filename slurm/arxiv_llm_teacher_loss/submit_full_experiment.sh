#!/bin/bash

set -eo pipefail

cd /hpc/home/jx198/iris_arxiv

EXPORT_JOB=$(sbatch --parsable \
    slurm/arxiv_llm_teacher_loss/export_all_chunks.sbatch)
SCORE_JOB=$(sbatch --parsable \
    --dependency="afterok:${EXPORT_JOB}" \
    slurm/arxiv_llm_teacher_loss/score_all_chunks_array.sbatch)
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${SCORE_JOB}" \
    slurm/arxiv_llm_teacher_loss/merge_teacher_scores.sbatch)
TRAIN_JOB=$(sbatch --parsable \
    --dependency="afterok:${MERGE_JOB}" \
    slurm/arxiv_llm_teacher_loss/train_variants.sbatch)
EVAL_JOB=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    slurm/arxiv_llm_teacher_loss/evaluate_variants.sbatch)
SUMMARY_JOB=$(sbatch --parsable \
    --dependency="afterok:${EVAL_JOB}" \
    slurm/arxiv_llm_teacher_loss/summarize_results.sbatch)

echo "Export job: ${EXPORT_JOB}"
echo "Qwen score array: ${SCORE_JOB}"
echo "Merge job: ${MERGE_JOB}"
echo "Training array: ${TRAIN_JOB}"
echo "Evaluation array: ${EVAL_JOB}"
echo "Summary job: ${SUMMARY_JOB}"
