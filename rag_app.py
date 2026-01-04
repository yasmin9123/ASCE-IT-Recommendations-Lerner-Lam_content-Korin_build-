"""
rag_app.py
Hierarchical RAG using:
- object_store/objects.jsonl (parents + children)
- Chroma (children only)
- SentenceTransformers embeddings
- OpenAI for answering

REQUIREMENTS:
- OPENAI_API_KEY set as environment variable
- folders:
    Downloads/
      rag_app.py
      object_store/
        objects.jsonl
"""

import os
import json
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
COLLECTION_NAME = "memo_children"
OBJECTS_PATH = os.path.join("object_store", "objects.jsonl")

TOP_K = 8
PARENT_EXPANSION = 3

# =========================
# SAFETY CHECK
# =========================

def answer_with_openai(query: str, context: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI()
    ...



# =========================
# OBJECT STORE LOADING
# =========================

def load_objects(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    objects = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objects.append(json.loads(line))
    return objects


def split_parents_children(objects: List[Dict[str, Any]]):
    parents = {}
    children = []

    for obj in objects:
        if obj.get("object_type") == "parent":
            parents[obj["id"]] = obj
        elif obj.get("object_type") == "child":
            children.append(obj)

    return parents, children


# =========================
# CHROMA SETUP
# =========================

def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def index_children(collection, embedder, children):
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


# =========================
# RETRIEVAL
# =========================

def retrieve_children(collection, embedder, query: str):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=TOP_K,
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
    for h in hits[:PARENT_EXPANSION]:
        pid = h["metadata"].get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)

    parts = []
    parts.append("=== RETRIEVED CHUNKS ===")
    for h in hits:
        parts.append(h["text"])
        parts.append("")

    parts.append("=== PARENT CONTEXT ===")
    for pid in parent_ids:
        if pid in parents:
            parts.append(parents[pid]["text"])
            parts.append("")

    return "\n".join(parts)


# =========================
# OPENAI CALL
# =========================

def answer_with_openai(query: str, context: str) -> str:
    client = OpenAI()

    messages = [
        {
            "role": "system",
            "content": "Answer using the provided context. If insufficient, say so."
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nContext:\n{context}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content


# =========================
# MAIN
# =========================

def main():
    print("Loading objects from object store...")
    objects = load_objects(OBJECTS_PATH)
    parents, children = split_parents_children(objects)
    print(f"Loaded {len(parents)} parents and {len(children)} children.")

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("Loading Chroma collection...")
    collection = get_collection()

    if collection.count() == 0:
        print("Indexing children into Chroma...")
        index_children(collection, embedder, children)
        print("Indexing complete.")
    else:
        print("Chroma index already exists.")

    print("\nReady. Type a question (or 'exit').\n")

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() in ("exit", "quit"):
            break

        hits = retrieve_children(collection, embedder, query)
        context = build_context(hits, parents)
        answer = answer_with_openai(query, context)

        print("\n=== ANSWER ===")
        print(answer)
        print("")


if __name__ == "__main__":
    main()
