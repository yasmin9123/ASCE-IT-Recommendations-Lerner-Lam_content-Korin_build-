"""
code.py — Hierarchical RAG (Object Store + Chroma + OpenAI)

What this does:
1) Reads object_store/objects.jsonl (parents + children)
2) Embeds ONLY child objects with a HuggingFace/SentenceTransformer model
3) Stores child embeddings in a persistent local Chroma DB
4) On each user query:
   - retrieves top child chunks from Chroma
   - expands context by pulling parent section(s) from the object store
   - calls OpenAI to answer using the retrieved context

You MUST set your OpenAI key as an environment variable:
  OPENAI_API_KEY=...

Windows (CMD):      setx OPENAI_API_KEY "your_key"
PowerShell:         setx OPENAI_API_KEY "your_key"
macOS/Linux (bash): export OPENAI_API_KEY="your_key"

Then restart your terminal/Python.
"""

import os
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from openai import OpenAI


# =========================
# 1) CONFIG (edit these)
# =========================

# Uses your environment variable (safe to share code)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# If you already used a different embedding model previously, use the SAME model here.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chroma persistent storage directory (local folder)
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "memo_children"

# Object store input file
OBJECTS_PATH = os.path.join("object_store", "objects.jsonl")


# Retrieval behavior
TOP_K_CHILDREN = 8              # how many child chunks to retrieve
EXPAND_PARENTS_FOR_TOP_N = 3    # how many top child hits trigger parent expansion
MAX_CONTEXT_CHARS = 22000       # basic guardrail to avoid overlong context payloads


# =========================
# 2) SAFETY CHECKS
# =========================

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set it as an environment variable and restart Python.\n"
        "Example (Windows CMD): setx OPENAI_API_KEY \"your_key\""
    )


# =========================
# 3) OBJECT STORE LOADING
# =========================

def load_objects_jsonl(objects_path: str) -> List[Dict[str, Any]]:
    """Load JSONL objects (one JSON object per line)."""
    if not os.path.exists(objects_path):
        raise FileNotFoundError(f"Could not find: {objects_path}\nCreate it and try again.")

    objects: List[Dict[str, Any]] = []
    with open(objects_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {objects_path} line {line_num}: {e}")

    return objects


def split_parents_children(objects: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      parents_by_id: dict of parent_id -> parent_object
      children: list of child objects
    """
    parents_by_id: Dict[str, Dict[str, Any]] = {}
    children: List[Dict[str, Any]] = []

    for obj in objects:
        obj_type = obj.get("object_type")
        if obj_type == "parent":
            if "id" not in obj:
                continue
            parents_by_id[obj["id"]] = obj
        elif obj_type == "child":
            # require id, parent_id, text
            if obj.get("id") and obj.get("parent_id") and obj.get("text"):
                children.append(obj)

    return parents_by_id, children


def build_children_fingerprint(children: List[Dict[str, Any]]) -> str:
    """
    Create a stable fingerprint of children so we can skip rebuilding embeddings
    if objects.jsonl didn't change.
    """
    # Only include stable fields
    packed = []
    for c in children:
        packed.append({
            "id": c.get("id"),
            "parent_id": c.get("parent_id"),
            "title": c.get("title"),
            "text": c.get("text"),
            "metadata": c.get("metadata", {})
        })
    blob = json.dumps(packed, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =========================
# 4) CHROMA SETUP / INDEXING
# =========================

def get_chroma_collection():
    """Create/load persistent Chroma collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def ensure_index_up_to_date(collection, embedder: SentenceTransformer, children: List[Dict[str, Any]]) -> None:
    """
    Indexing strategy:
    - We store a single "index_meta" record with a fingerprint.
    - If fingerprint matches, we skip re-embedding.
    - If fingerprint differs, we rebuild the collection.
    """
    fingerprint = build_children_fingerprint(children)

    # Try to read index meta
    try:
        meta_get = collection.get(ids=["__index_meta__"], include=["metadatas"])
        existing_meta = meta_get["metadatas"][0] if meta_get and meta_get.get("metadatas") else None
        existing_fp = existing_meta.get("fingerprint") if existing_meta else None
    except Exception:
        existing_fp = None

    if existing_fp == fingerprint:
        print("Chroma index is up to date (no re-embedding needed).")
        return

    # Rebuild: easiest + safest for beginners
    print("Objects changed (or index missing). Rebuilding Chroma index...")

    # Delete all existing entries by recreating the collection
    # (Chroma doesn't have 'drop collection' directly from this object, so we re-init client)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    upsert_children(collection, embedder, children)

    # Save index meta record (no embedding needed)
    collection.upsert(
        ids=["__index_meta__"],
        documents=["index metadata record"],
        metadatas=[{"fingerprint": fingerprint}]
    )

    print("Rebuild complete.")


def upsert_children(collection, embedder: SentenceTransformer, children: List[Dict[str, Any]]) -> None:
    """Embed and store child objects in Chroma."""
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for child in children:
        cid = child["id"]
        text = child.get("text", "").strip()
        if not text:
            continue

        meta = dict(child.get("metadata", {}))
        meta["doc_id"] = child.get("doc_id")
        meta["parent_id"] = child.get("parent_id")
        meta["title"] = child.get("title")

        ids.append(cid)
        docs.append(text)
        metas.append(meta)

    if not ids:
        raise RuntimeError("No child objects found to embed/upsert. Check your objects.jsonl.")

    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas
    )

    print(f"Upserted {len(ids)} child objects into '{CHROMA_COLLECTION}'.")


# =========================
# 5) RETRIEVAL + HIERARCHICAL EXPANSION
# =========================

def retrieve_children(collection, embedder: SentenceTransformer, query: str, top_k: int) -> List[Dict[str, Any]]:
    """Retrieve top child chunks from Chroma by semantic similarity."""
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances", "ids"]
    )

    hits: List[Dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    # Remove the meta record if it ever shows up (it shouldn't, but safe)
    hits = [h for h in hits if h["id"] != "__index_meta__"]
    return hits


def build_context(
    query: str,
    hits: List[Dict[str, Any]],
    parents_by_id: Dict[str, Dict[str, Any]],
    expand_parents_for_top_n: int,
    max_chars: int
) -> str:
    """
    Build a RAG context with:
    - top retrieved child chunks (always)
    - expanded parent section(s) for top N hits (deduped)
    Includes chunk IDs so the model can cite them.
    """
    hits_sorted = sorted(hits, key=lambda x: x["distance"])

    # Determine which parents to expand (top N hits only)
    expanded_parent_ids: List[str] = []
    for h in hits_sorted[:expand_parents_for_top_n]:
        pid = h["metadata"].get("parent_id")
        if pid and pid not in expanded_parent_ids:
            expanded_parent_ids.append(pid)

    parts: List[str] = []
    parts.append("INSTRUCTIONS: Use the memo context below. Prefer grounded answers. Cite chunk IDs when relevant.")
    parts.append("")

    parts.append("=== TOP RETRIEVED CHUNKS (children) ===")
    for rank, h in enumerate(hits_sorted, start=1):
        md = h["metadata"] or {}
        parts.append(f"[Child {rank}] id={h['id']} section={md.get('section')} type={md.get('type')} title={md.get('title')}")
        parts.append(h["text"])
        parts.append("")

    parts.append("=== EXPANDED SECTION CONTEXT (parents) ===")
    for pid in expanded_parent_ids:
        parent = parents_by_id.get(pid)
        if not parent:
            continue
        parts.append(f"[Parent] id={pid} title={parent.get('title')}")
        parts.append(parent.get("text", "").strip())
        parts.append("")

    context = "\n".join(parts).strip()

    # Guardrail: trim if too long
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[Context trimmed for length.]"

    return context


# =========================
# 6) OPENAI ANSWERING
# =========================

def answer_with_openai(query: str, context: str) -> str:
    """
    Uses OpenAI SDK with API key pulled from environment automatically.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer using ONLY the provided memo context when possible. "
                "If the context is insufficient, say so. When you rely on content, cite chunk IDs (e.g., C4_2, C5_F2)."
            )
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"
        }
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )

    return resp.choices[0].message.content


# =========================
# 7) MAIN: BUILD INDEX ONCE + QUERY LOOP
# =========================

def main():
    print("Loading objects from object store...")
    objects = load_objects_jsonl(OBJECTS_PATH)
    parents_by_id, children = split_parents_children(objects)
    print(f"Loaded {len(parents_by_id)} parents and {len(children)} children.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading Chroma collection...")
    collection = get_chroma_collection()

    # Ensure index matches current objects.jsonl
    ensure_index_up_to_date(collection, embedder, children)

    print("\nReady. Type a question (or type 'exit').\n")

    while True:
        query = input("Enter your question: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        hits = retrieve_children(collection, embedder, query, top_k=TOP_K_CHILDREN)

        if not hits:
            print("\nNo results retrieved from the vector store.\n")
            continue

        context = build_context(
            query=query,
            hits=hits,
            parents_by_id=parents_by_id,
            expand_parents_for_top_n=EXPAND_PARENTS_FOR_TOP_N,
            max_chars=MAX_CONTEXT_CHARS
        )

        answer = answer_with_openai(query, context)

        print("\n=== ANSWER ===")
        print(answer)
        print("")


if __name__ == "__main__":
    main()
