# IRIS: efficient infinite context retrieval-based classification

## Development installation

```bash
conda env create -f environment.yml
conda activate iris
git clone https://github.com/fengnanli-neo/iris.git
cd iris
pip install -e .
```

## Usage

To change the program behavior, modify the parameters in `config.toml`, then run

```bash
python -m iris
```

## arXiv source-text traceback

This branch extends IRIS so that every retrieved vector chunk can be traced back to its original arXiv text.

Each chunk stored in LanceDB contains:

- `id`: internal document ID
- `chunk_id`: chunk number within the document
- `offset`: starting token position
- `label`: document category
- `source_path`: path to the original `.txt` file
- `chunk_text`: exact text taken from the original document
- `vector`: ModernBERT embedding

The original chunk text is extracted using tokenizer character offsets. It is not reconstructed by decoding token IDs.

### Generate synthetic arXiv data

```bash
python scripts/generate_fake_arxiv.py
```

### Build the synthetic dataset and vector database

```bash
PYTHONPATH=src python -m iris \
  --config config.arxiv.fake.toml
```

### Display top-k chunks and their original text

```bash
PYTHONPATH=src python scripts/show_arxiv_topk.py \
  --config config.arxiv.fake.toml \
  --split test
```

The generated datasets, vector databases, and model checkpoints are excluded from Git.