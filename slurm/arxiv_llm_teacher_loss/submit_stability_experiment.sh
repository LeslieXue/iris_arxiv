#!/bin/bash

set -eo pipefail

cd /hpc/home/jx198/iris_arxiv

TRAIN_JOB=$(sbatch --parsable \
    slurm/arxiv_llm_teacher_loss/train_stability_variants.sbatch)
EVAL_JOB=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    slurm/arxiv_llm_teacher_loss/evaluate_stability_variants.sbatch)
SUMMARY_JOB=$(sbatch --parsable \
    --dependency="afterok:${EVAL_JOB}" \
    slurm/arxiv_llm_teacher_loss/summarize_stability_results.sbatch)

echo "Training array (10 lambdas x 5 seeds): ${TRAIN_JOB}"
echo "Evaluation array: ${EVAL_JOB}"
echo "Summary job: ${SUMMARY_JOB}"
