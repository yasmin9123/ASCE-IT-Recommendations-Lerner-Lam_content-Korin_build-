import os
import json
import hashlib
from typing import Dict, List, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================
# CONFIG
# =========================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION = "memo_children"
OBJECTS_PATH = os.path.join("object_store", "objects.jsonl")

TOP_K_CHILDREN = 8
EXPAND_PARENTS_FOR_TOP_N = 3
MAX_CONTEXT_CHARS = 20000


# =========================
# SAFETY CHECK
# =========================

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "Set it as an environment variable and restart the terminal."
    )


# =========================
# OBJECT STORE
# =========================

def load_objects_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    objects = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return objects


def split_parents_children(objects):
    parents = {}
    children = []

    for obj in objects:
        if obj.get("object_type") == "parent":
            parents[obj["id"]] = obj
        elif obj.get("object_type") == "child":
            children.append(obj)

    return parents, children


def fingerprint_children(children):
    blob = json.dumps(
        [
            {
                "id": c["id"],
                "parent_id": c["parent_id"],
                "text": c["text"]
            }
            for c in children
        ],
        sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =========================
# CHROMA
# =========================

def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def rebuild_index_if_needed(collection, embedder, children):
    fp = fingerprint_children(children)

    try:
        meta = collection.get(ids=["__meta__"], include=["metadatas"])
        old_fp = meta["metadatas"][0].get("fingerprint")
    except Exception:
        old_fp = None

    if fp == old_fp:
        print("Chroma index is up to date (no re-embedding needed).")
        return

    print("Rebuilding Chroma index...")

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

    ids = []
    docs = []
    metas = []

    for c in children:
        ids.append(c["id"])
        docs.append(c["text"])
        meta = dict(c.get("metadata", {}))
        meta["parent_id"] = c["parent_id"]
        meta["title"] = c.get("title")
        metas.append(meta)

    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metas
    )

    collection.upsert(
        ids=["__meta__"],
        documents=["index metadata"],
        metadatas=[{"fingerprint": fp}]
    )

    print(f"Upserted {len(ids)} child objects into '{CHROMA_COLLECTION}'.")


# =========================
# RETRIEVAL
# =========================

def retrieve_children(collection, embedder, query, top_k):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return hits


def build_context(hits, parents):
    hits = sorted(hits, key=lambda x: x["distance"])

    parent_ids = []
    for h in hits[:EXPAND_PARENTS_FOR_TOP_N]:
        pid = h["metadata"].get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)

    parts = []
    parts.append("=== RETRIEVED CHUNKS ===\n")

    for i, h in enumerate(hits, 1):
        parts.append(
            f"[Chunk {i}] "
            f"section={h['metadata'].get('section')} "
            f"type={h['metadata'].get('type')}"
        )
        parts.append(h["text"])
        parts.append("")

    parts.append("\n=== PARENT CONTEXT ===\n")

    for pid in parent_ids:
        p = parents.get(pid)
        if p:
            parts.append(f"[Section] {p.get('title')}")
            parts.append(p.get("text", ""))
            parts.append("")

    context = "\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]


# =========================
# OPENAI
# =========================

def answer_with_openai(query, context):
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer using ONLY the provided memo context when possible."
            )
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nContext:\n{context}"
        }
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )

    return resp.choices[0].message.content


# =========================
# MAIN
# =========================

def main():
    print("Loading objects from object store...")
    objects = load_objects_jsonl(OBJECTS_PATH)
    parents, children = split_parents_children(objects)
    print(f"Loaded {len(parents)} parents and {len(children)} children.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading Chroma collection...")
    collection = get_collection()

    rebuild_index_if_needed(collection, embedder, children)

    print("\nReady. Type a question (or type 'exit').\n")

    while True:
        query = input("Enter your question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        hits = retrieve_children(collection, embedder, query, TOP_K_CHILDREN)
        context = build_context(hits, parents)
        answer = answer_with_openai(query, context)

        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n")


if __name__ == "__main__":
    main()
