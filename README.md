# Vector Db Benchmark

Benchmark FAISS vs simple NumPy cosine search on synthetic embeddings (latency/recall).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  CLI / API  │ ──► │ Core logic   │ ──► │  Models /   │
│  entrypoint │     │  (src/)      │     │  stores     │
└─────────────┘     └──────────────┘     └─────────────┘
```
Modules are intentionally small so you can read the full control flow in one sitting.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --help
```

## Project layout

```
src/          # application code
data/         # sample inputs (optional)
tests/        # smoke tests
README.md     # this file
```

## Notes

Educational MVP — intended for learning architecture and extending into production.
