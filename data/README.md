# Data files

The application expects processed judgment data at:

- `data/processed/chunks.jsonl`
- `data/processed/judgments.jsonl`
- `data/processed/qdrant_manifest.json`

The judgment corpus, raw archives, validation datasets, and generated vector
caches are intentionally excluded from the Git repository because they are
large, may have separate distribution terms, and can exceed GitHub's normal
file-size limit. Obtain or generate the datasets separately before running
the full retrieval pipeline. The manifest is kept in Git as a reference for
the Qdrant collection configuration.
