# Mini-CLIP — runnable workbench

Inspect cosine retrieval, compare candidates and understand the embedding interface.

## Open the interface

Python 3.10+; no npm install, API keys, model download or GPU required for the demo.

```sh
python serve_demo.py
```

Open <http://127.0.0.1:8000>. To use another port: `python serve_demo.py --port 8001`.
Serve `demo/` over HTTP rather than opening `index.html` as a file. The server binds only to localhost.

## What is implemented

`src/retrieval.py` searches and persists normalized embeddings. Supply an NPZ index created with `EmbeddingIndex(vectors, labels).save(path)` and a query from the same encoder space. The browser uses hand-authored three-dimensional illustrative vectors; it does not demonstrate trained retrieval quality. The command below assumes you created `embeddings.npz`.

Interface capabilities: Cosine search, Embedding explorer, Reusable Python index.

## Core workflow

From the repository root (install the project's existing dependencies first for model-backed commands):

```sh
python -m src.retrieval embeddings.npz --query "[1, 0, 0]" -k 3
```

## Verify

The browser's calculation tests use Node 18+ and no third-party packages:

```sh
node --test demo/core.test.cjs
python -m pytest tests/test_retrieval.py tests/test_training_regressions.py -q
```

The GitHub demo workflow checks browser calculations and the dependency-free core tests where present. Full original model/service suites have separate dependencies; passing demo checks does not certify GPU training or external services.

## Before live use

Real retrieval quality requires a trained checkpoint and exported encoder embeddings.

Local browser data belongs to this browser profile. Export records before clearing it or changing devices. Photo previews stay in memory; exported briefs contain filenames, not uploaded image bytes. There are no payment or outbound model calls in this demo.

## Provenance

Workbench and starter additions: AI-assisted implementation, 2026-11-13, for Ray's review. Existing repository history and upstream attribution are preserved. New starter directions are proposals where the GitHub repository had no code.
