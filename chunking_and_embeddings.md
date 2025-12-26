# Chunking and Embeddings (Practical Guidance)

Chunking
- Prefer paragraph- or section-level chunks over fixed-length slicing.
- Preserve headings in chunk text to keep governance context.
- Use overlap (10–20%) to avoid boundary losses.

Embeddings
- Choose one embedding model and record it in every vector record.
- Keep embeddings immutable for a given token version; if you change models, version the token.

Metadata (minimum)
- object_type (governance_policy, ethics_guidance, technical_note, etc.)
- governance_tags (e.g., "draft", "reviewed", "validated", "gold_seal")
- attribution_tags (e.g., author, committee, publication year)
- policy_gate_refs (IDs of gates that apply)

Top-k
- “Top-k” means return the k most similar vectors (highest similarity score) after filtering by policy gates.
