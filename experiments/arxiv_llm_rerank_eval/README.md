# arXiv LLM Reranking Evaluation

This experiment evaluates whether an LLM teacher can improve the
explanatory quality of text chunks retrieved by IRIS.

The experiment compares:

1. Original IRIS vector-distance ranking
2. LLM-only reranking
3. Hybrid IRIS and LLM reranking
4. Shuffled-score control

Each IRIS query retrieves eight candidate chunks. Each method retains
four chunks per query, preserving the original maximum budget of
32 chunks per document.

## Label definitions

- `cs.AI`: Artificial Intelligence
- `cs.DS`: Data Structures and Algorithms
- `cs.PL`: Programming Languages

## Leakage controls

The teacher LLM must never receive:

- the true article label
- the source path
- the vector distance
- the original IRIS rank
- the retrieval method name

Generated candidates, scores, rankings, and evaluation results are
stored under `/work` and are not committed to Git.
