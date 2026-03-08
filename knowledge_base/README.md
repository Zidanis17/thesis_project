# Knowledge Base

Place retrieval documents for the AV ethics RAG layer in the subfolders here.

Supported file types:
- `.md`
- `.txt`
- `.json`
- `.pdf`

Recommended layout:
- `german_ethics_commission/`
- `ethical_frameworks/`
- `legal_constraints/`
- `similar_scenarios/`

Ingestion is separate from runtime retrieval:
- Offline ingestion reads these files, extracts PDF text, chunks documents, and stores embeddings in Chroma.
- Runtime retrieval only queries the persisted Chroma collection.
