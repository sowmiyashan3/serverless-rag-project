import faiss
import json
import os
import threading
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from azure.storage.blob import BlobServiceClient


# ---------------- CONFIG ----------------
INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
ID_MAP_PATH = os.path.join(INDEX_DIR, "id_mapping.json")

# Azure Blob config
BLOB_CONN_STR = "<Your-Connection-String>"
BLOB_CONTAINER = "wiki-metadata"

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER)

EMBEDDING_DIM = 384  # change based on model
LOCK = threading.Lock()

# ---------------- MODELS ----------------
class EmbeddingRequest(BaseModel):
    chunk_id: str
    embedding: List[float]

class SearchRequest(BaseModel):
    query_embedding: List[float]
    top_k: int = 5

class EmbeddingBatchRequest(BaseModel):
    items: List[EmbeddingRequest]


# ---------------- APP ----------------
app = FastAPI(title="FAISS Vector Service")

# ---------------- LOAD / INIT ----------------
def load_or_create_index():
    if os.path.exists(INDEX_PATH):
        print("🔁 Loading FAISS index from disk...")
        index = faiss.read_index(INDEX_PATH)
        with open(ID_MAP_PATH, "r") as f:
            id_map = json.load(f)
    else:
        print("🆕 Creating new FAISS index...")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        id_map = {}

    return index, id_map

index, id_mapping = load_or_create_index()

# ---------------- HELPERS ----------------
def persist_index():
    faiss.write_index(index, INDEX_PATH)
    with open(ID_MAP_PATH, "w") as f:
        json.dump(id_mapping, f)


def get_chunk_text_from_blob(chunk_id: str):
    """
    Search JSONL files in blob for the chunk_id and return chunk_text
    """
    blobs = container_client.list_blobs(name_starts_with="documents_chunks")  # or a prefix
    for blob in blobs:
        blob_client = container_client.get_blob_client(blob)
        data = blob_client.download_blob().readall()

        lines = data.decode("utf-8").splitlines()
        for line in lines:
            item = json.loads(line)
            if item["chunk_index"] is not None and f'{item["doc_id"]}_{item["chunk_index"]}' == chunk_id:
                return item["chunk_text"]
    return "[CONTENT NOT FOUND]"

# ---------------- ROUTES ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectors": index.ntotal,
        "index_loaded": True
    }

@app.post("/add")
def add_embeddings_batch(req: EmbeddingBatchRequest):
    # Validate embeddings
    for item in req.items:
        if len(item.embedding) != EMBEDDING_DIM:
            raise HTTPException(400, f"Invalid embedding size for chunk_id {item.chunk_id}")
    
    vectors_to_add = []
    new_ids = []
    
    with LOCK:
        for item in req.items:
            if item.chunk_id in id_mapping:
                continue  # skip duplicates
            vectors_to_add.append(item.embedding)
            new_ids.append(item.chunk_id)

        if not vectors_to_add:
            return {"status": "skipped", "reason": "all embeddings already indexed"}

        # Convert batch to numpy array and normalize
        vectors = np.array(vectors_to_add, dtype="float32")
        faiss.normalize_L2(vectors)
        index.add(vectors)

        # Update id mapping
        start_idx = index.ntotal - len(vectors)
        for i, chunk_id in enumerate(new_ids):
            id_mapping[chunk_id] = start_idx + i

        persist_index()

    return {"status": "indexed", "total_vectors": index.ntotal, "added": len(new_ids)}



@app.post("/search")
def search(req: SearchRequest):
    query = np.array([req.query_embedding]).astype("float32")
    faiss.normalize_L2(query)


    scores, ids = index.search(query, req.top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk_id = list(id_mapping.keys())[list(id_mapping.values()).index(idx)]
        chunk_text = get_chunk_text_from_blob(chunk_id)
        
        results.append({
            "chunk_id": chunk_id,
            "score": float(score),
            "content":  chunk_text
        })

    return {"results": results}

