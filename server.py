def index_children(collection, embedder, children):
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

def build_parent_context(hits, parents):
    hits = sorted(hits, key=lambda x: x["distance"])
    parent_ids = []
    for h in hits[:PARENT_EXPANSION]:
        pid = h["metadata"].get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)

    parts = []
    for pid in parent_ids:
        if pid in parents:
            parts.append(parents[pid]["text"])
            parts.append("")
    return "\n".join(parts).strip()

@app.on_event("startup")
def startup():
    global _embedder, _collection, _parents
    objects = load_objects(OBJECTS_PATH)
    parents, children = split_parents_children(objects)
    _parents = parents

    _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    _collection = get_collection()
    index_children(_collection, _embedder, children)

@app.get("/health")
def health():
    return {"ok": True, "collection": COLLECTION_NAME, "objects_path": OBJECTS_PATH}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    try:
        hits = retrieve_children(_collection, _embedder, req.query, req.top_k)
        parent_context = build_parent_context(hits, _parents)
        return SearchResponse(
            query=req.query,
            top_k=req.top_k,
            hits=[SearchHit(**h) for h in hits],
            parent_context=parent_context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
