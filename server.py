# server.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# We reuse your existing RAG logic + constants from rag_app.py
# This assumes rag_app.py is in the same repo root.
import rag_app


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="RAG Search API", version="1.0")


# -----------------------------
# Config
# -----------------------------
TOP_K_DEFAULT = 5
PARENT_EXPANSION = 3  # how many top hits to expand into parent context


# -----------------------------
# Response / Request Models
# -----------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(TOP_K_DEFAULT, ge=1, le=50)


class SearchHit(BaseModel):
    text: str
    metadata: Dict[str, Any]
    distance: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[SearchHit]
    parent_context: str


# -----------------------------
# Globals initialized at startup
# -----------------------------
_embedder = None
_collection = None
_parents: Optional[Dict[str, Any]] = None


# -----------------------------
# Core helpers (your logic, fixed)
# -----------------------------
def index_children(collection, embedder, children: List[Dict[str, Any]]) -> None:
    ids, docs, metas = [], [], []
    for c in children:
        ids.append(c["id"])
        docs.append(c["text"])
        meta = dict(c.get("metadata", {}))
        meta["parent_id"] = c["parent_id"]
        meta["title"] = c.get("title")
        metas.append(meta)

    embeddings = embedder.encode(docs, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)


def retrieve_children(collection, embedder, query: str, top_k: int):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    # results fields are nested lists: results["documents"][0] is the list of docs
    for i in range(len(results["documents"][0])):
        hits.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits


def build_parent_context(hits, parents: Dict[str, Any]) -> str:
    hits = sorted(hits, key=lambda x: x["distance"])
    parent_ids: List[str] = []
    for h in hits[:PARENT_EXPANSION]:
        pid = h["metadata"].get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)

    parts: List[str] = []
    for pid in parent_ids:
        if pid in parents:
            parts.append(parents[pid]["text"])
            parts.append("")  # blank line between parents
    return "\n".join(parts).strip()


# -----------------------------
# Startup: load objects, embedder, collection
# -----------------------------
@app.on_event("startup")
def startup():
    global _embedder, _collection, _parents

    # Load your object store
    objects = rag_app.load_objects(rag_app.OBJECTS_PATH)
    parents, children = rag_app.split_parents_children(objects)
    _parents = parents

    # Load embedding model + chroma collection
    _embedder = rag_app.SentenceTransformer(rag_app.EMBED_MODEL_NAME)
    _collection = rag_app.get_collection()

    # IMPORTANT: only index if empty, so you don't duplicate on every restart
    if _collection.count() == 0:
        index_children(_collection, _embedder, children)


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "collection": rag_app.COLLECTION_NAME,
        "objects_path": rag_app.OBJECTS_PATH,
        "count": _collection.count() if _collection is not None else None,
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        if _collection is None or _embedder is None or _parents is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        hits = retrieve_children(_collection, _embedder, req.query, req.top_k)
        parent_context = build_parent_context(hits, _parents)

        return SearchResponse(
            query=req.query,
            top_k=req.top_k,
            hits=[SearchHit(**h) for h in hits],
            parent_context=parent_context,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
