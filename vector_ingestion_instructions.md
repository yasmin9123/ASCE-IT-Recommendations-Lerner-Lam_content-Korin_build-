# Vector Store Ingestion Instructions (Template)

Acronyms
- Vector Store: database optimized for similarity search over embeddings
- Object Store: authoritative repository for canonical artifacts
- V&V: Verification and Validation
- Model Context Protocols (MCPs): conventions for passing structured context into AI runtimes

Overview
1. Select which Object Store artifacts will be searchable (typically: governance notes, policies, technical memos).
2. Convert each artifact to canonical text for indexing (retain the original in the Object Store).
3. Chunk text into semantically coherent units (e.g., 400–800 tokens with overlap).
4. Generate embeddings using a stable embedding model.
5. Create one vector record per chunk using `02_schemas/vector_record.schema.json`.
6. Store vector records in your Vector Store. Each record must include:
   - token_id, object_id, hash references
   - governance_tags and attribution_tags
   - policy_gate_refs to enforce retrieval-time constraints

Retrieval-time behavior (recommended)
- For every user prompt, attach policy gates to the retrieval request and filter vector records accordingly.
- Return attribution metadata with any retrieved content.
- Log access for auditability, especially for “reviewed/validated/gold_seal” tiers.
