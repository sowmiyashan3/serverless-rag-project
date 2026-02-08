import os
import json
import logging
import threading
from typing import List, Set, Dict, Optional
from pydantic import BaseModel
from threading import Lock

from fastapi import FastAPI, HTTPException
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# ---------------- CONFIG ----------------
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
OPENSEARCH_INDEX = os.environ["OPENSEARCH_INDEX"]
OPENSEARCH_KEY = os.environ["OPENSEARCH_KEY"]

STATE_DIR = "state"
LOG_DIR = "logs"

os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

INDEXED_IDS_PATH = os.path.join(STATE_DIR, "indexed_ids.json")

# ---------------- APP ----------------
app = FastAPI(title="OpenSearch Ingestion Service")
lock = Lock()

# ---------------- LOGGING ----------------
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "opensearch.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(OPENSEARCH_ENDPOINT)
logger.info(OPENSEARCH_INDEX)
logger.info(OPENSEARCH_KEY)

# ---------------- OPENSEARCH CLIENT ----------------
search_client = SearchClient(
    endpoint=OPENSEARCH_ENDPOINT,
    index_name=OPENSEARCH_INDEX,
    credential=AzureKeyCredential(OPENSEARCH_KEY)
)

# ---------------- STATE ----------------
def load_indexed_ids() -> Set[str]:
    if os.path.exists(INDEXED_IDS_PATH):
        with open(INDEXED_IDS_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_indexed_ids(ids: Set[str]):
    with open(INDEXED_IDS_PATH, "w") as f:
        json.dump(sorted(ids), f)

indexed_ids = load_indexed_ids()


# ---------------- PAYLOAD MODELS ----------------
class ChunkItem(BaseModel):
    chunk_id: str
    chunk_text: str
    metadata: Dict = {}

class Payload(BaseModel):
    items: List[ChunkItem]

class SearchResponseItem(BaseModel):
    id: str
    content: str
    metadata: Dict = {}

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[SearchResponseItem]

# ---------------- ROUTES ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_docs": len(indexed_ids)
    }

@app.post("/add")
def index_chunks(payload: Payload):
    """
    Accepts payload of the form:
    {
        "items": [
            {
                "chunk_id": str,
                "chunk_text": str,
                "metadata": {...}
            },
            ...
        ]
    }
    """
    to_index = []

    with lock:
        for item in payload.items:
            if item.chunk_id in indexed_ids:
                continue

            doc = {
                "id": item.chunk_id,
                "content": item.chunk_text,
                **item.metadata
            }

            to_index.append(doc)

        if not to_index:
            return {"status": "skipped", "reason": "all chunks already indexed"}

        try:
            # Upload in batch to OpenSearch
            search_client.upload_documents(documents=to_index)
        except Exception as e:
            logger.exception("OpenSearch indexing failed")
            raise HTTPException(status_code=500, detail=str(e))

        # Update indexed IDs
        for doc in to_index:
            indexed_ids.add(doc["id"])

        save_indexed_ids(indexed_ids)

    logger.info(f"Indexed {len(to_index)} documents")
    return {
        "status": "indexed",
        "count": len(to_index),
        "total_indexed": len(indexed_ids)
    }


@app.get("/search", response_model=SearchResponse)
def search_docs(q: str, top_k: Optional[int] = 5):
    """
    Search indexed documents.
    Example: GET /search?q=machine+learning&top_k=5
    """
    try:
        results = list(
            search_client.search(
                search_text=q,
                top=top_k,
                select=["id", "content"]
            )
        )

        # ----------------------------
        # 1️⃣ Collect raw scores
        # ----------------------------
        raw_scores = [
            r["@search.score"] for r in results if "@search.score" in r
        ]
        max_score = 100

        # ----------------------------
        # 2️⃣ Build response with normalized score
        # ----------------------------
        response_items = []
        for r in results:
            raw_score = r.get("@search.score", 0.0)
            normalized_score = raw_score / max_score  # 👈 KEY LINE

            response_items.append(
                SearchResponseItem(
                    id=r["id"],
                    content=r.get("content", ""),
                    metadata={
                        **{k: v for k, v in r.items() if k not in ["id", "content"]},
                        "raw_score": raw_score,
                        "normalized_score": normalized_score
                    }
                )
            )

        return SearchResponse(
            query=q,
            total_results=len(response_items),
            results=response_items
        )

    except Exception as e:
        logger.exception("Azure AI Search query failed")
        raise HTTPException(status_code=500, detail=str(e))





