import os
import json
import logging
import time
from typing import Set, List, Dict, Any
import gc

import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from fastembed import TextEmbedding
import numpy as np

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
STORAGE_CONN = os.environ["WikiStorageConnection"]
FAISS_ENDPOINT = os.environ["FaissEndpoint"]
OPENSEARCH_ENDPOINT = os.environ["OpensearchEndpoint"]

METADATA_CONTAINER = "wiki-metadata"
CHUNKS_FILE = "documents_chunks.jsonl"

# =========================================================
# DIRECTORIES & STATE
# =========================================================
STATE_DIR = "state"
LOG_DIR = "logs"
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

EMBEDDED_IDS_PATH = os.path.join(STATE_DIR, "embedded_ids.json")

# =========================================================
# LOGGING SETUP
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "embedding.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================
BATCH_SIZE = 64
WRITE_BUFFER_SIZE = 8000
EMBED_DIM = 384
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# =========================================================
# GLOBAL STATE
# =========================================================
embedded_ids: Set[str] = set()
is_ingesting = False
ingestion_stats = {
    "status": "idle",
    "total_processed": 0,
    "total_embedded": 0,
    "faiss_added": 0,
    "opensearch_indexed": 0,
    "current_rate": 0.0,
    "start_time": None,
    "last_update": None,
    "error": None
}

# =========================================================
# LOAD/SAVE EMBEDDED IDS
# =========================================================
def load_embedded_ids() -> Set[str]:
    """Load set of already-processed chunk IDs"""
    if os.path.exists(EMBEDDED_IDS_PATH):
        try:
            with open(EMBEDDED_IDS_PATH, "r") as f:
                return set(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load embedded_ids: {e}")
            return set()
    return set()

def save_embedded_ids(ids: Set[str]):
    """Atomically save embedded IDs"""
    try:
        tmp = EMBEDDED_IDS_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(ids), f)
        os.replace(tmp, EMBEDDED_IDS_PATH)
        logger.debug(f"Saved {len(ids)} embedded IDs")
    except Exception as e:
        logger.error(f"Error saving embedded IDs: {e}")

# Load existing state on startup
embedded_ids = load_embedded_ids()
logger.info(f"Loaded {len(embedded_ids)} previously embedded chunks")


# =========================================================
# STREAM CHUNKS FROM AZURE BLOB
# =========================================================
def stream_chunks_from_blob():
    """Stream chunks line-by-line from Azure Blob Storage"""
    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONN)
    container = blob_service.get_container_client(METADATA_CONTAINER)
    blob = container.get_blob_client(CHUNKS_FILE)
    
    logger.info(f"Streaming chunks from {CHUNKS_FILE}")
    downloader = blob.download_blob()
    buffer = ""
    
    for chunk in downloader.chunks():
        buffer += chunk.decode('utf-8')
        lines = buffer.split('\n')
        buffer = lines[-1]
        
        for line in lines[:-1]:
            if line.strip():
                yield json.loads(line)
    
    if buffer.strip():
        yield json.loads(buffer)

# =========================================================
# LOAD FASTEMBED MODEL
# =========================================================
logger.info("Loading FastEmbed model...")
model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_length=256,
    threads=2
)
logger.info("✅ FastEmbed model loaded (INT8 quantized)")

# =========================================================
# EMBEDDING FUNCTION
# =========================================================
def embed_batch(texts: List[str], ids: List[str]) -> tuple[List[str], np.ndarray]:
    """Generate embeddings for a batch of texts"""
    try:
        embeddings_list = list(model.embed(texts))
        embeddings_np = np.vstack(embeddings_list).astype(np.float32)
        
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / (norms + 1e-8)
        
        return ids, embeddings_np
        
    except Exception as e:
        logger.error(f"Error embedding batch: {e}", exc_info=True)
        return [], np.array([])


# =========================================================
# SEND TO FAISS (BULK)
# =========================================================
def send_to_faiss_bulk(chunk_ids: List[str], embeddings: np.ndarray) -> bool:
    """Send large batch of embeddings to FAISS service"""
    try:
        items = [
            {"chunk_id": cid, "embedding": emb.tolist()}
            for cid, emb in zip(chunk_ids, embeddings)
        ]
        
        payload = {"items": items}
        
        logger.info(f"📤 Sending {len(chunk_ids):,} vectors to FAISS...")
        
        response = requests.post(
            f"{FAISS_ENDPOINT}",
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ FAISS: {result.get('added', 0):,} vectors added")
            return True
        else:
            logger.error(f"❌ FAISS error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ FAISS request failed: {e}", exc_info=True)
        return False

# =========================================================
# SEND TO OPENSEARCH (BULK)
# =========================================================
def send_to_opensearch_bulk(chunk_ids: List[str], chunks: List[dict]) -> bool:
    """Send large batch of documents to OpenSearch service"""
    try:
        items = [
            {
                "chunk_id": cid,
                "chunk_text": c["chunk_text"],
                "metadata": {
                    "doc_id": c["doc_id"],
                    "chunk_index": c["chunk_index"]
                }
            }
            for cid, c in zip(chunk_ids, chunks)
        ]
        
        payload = {"items": items}
        
        logger.info(f"📤 Sending {len(chunk_ids):,} documents to OpenSearch...")
        
        response = requests.post(
            f"{OPENSEARCH_ENDPOINT}",
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ OpenSearch: {result.get('count', 0):,} documents indexed")
            return True
        else:
            logger.error(f"❌ OpenSearch error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ OpenSearch request failed: {e}", exc_info=True)
        return False

# =========================================================
# MAIN INGESTION FUNCTION (WITH DETAILED TIMING)
# =========================================================
def run_ingestion():
    """Main ingestion loop with detailed timing diagnostics"""
    global embedded_ids, is_ingesting, ingestion_stats
    
    if is_ingesting:
        logger.warning("Ingestion already in progress")
        return
    
    is_ingesting = True
    ingestion_stats["status"] = "running"
    ingestion_stats["start_time"] = time.time()
    ingestion_stats["error"] = None
    
    chunks_buffer = []
    write_buffer_ids = []
    write_buffer_embeddings = []
    write_buffer_chunks = []
    
    total_new = 0
    total_faiss_success = 0
    total_opensearch_success = 0
    start_time = time.time()
    
    # ✅ TIMING TRACKERS
    total_stream_time = 0
    total_embed_time = 0
    total_faiss_time = 0
    total_opensearch_time = 0
    chunks_streamed = 0
    
    logger.info("="*60)
    logger.info("Starting ingestion with detailed timing")
    logger.info(f"Embedding batch size: {BATCH_SIZE}")
    logger.info(f"Write buffer size: {WRITE_BUFFER_SIZE:,}")
    logger.info(f"Already processed: {len(embedded_ids):,} chunks")
    logger.info("="*60)

    try:
        stream_start = time.time()
        
        for chunk in stream_chunks_from_blob():
            chunks_streamed += 1
            cid = f"{chunk['doc_id']}_{chunk['chunk_index']}"
            
            # Skip if already processed
            if cid in embedded_ids:
                continue
            
            chunks_buffer.append((chunk["chunk_text"], cid, chunk))
            
            # Process embedding batch
            if len(chunks_buffer) >= BATCH_SIZE:
                # Track streaming time
                stream_time = time.time() - stream_start
                total_stream_time += stream_time
                
                texts, ids, chunks = zip(*chunks_buffer)
                
                # Generate embeddings (TIMED)
                embed_start = time.time()
                batch_ids, embeddings = embed_batch(list(texts), list(ids))
                embed_time = time.time() - embed_start
                total_embed_time += embed_time
                
                if len(batch_ids) == 0:
                    logger.warning("Embedding batch failed, skipping")
                    chunks_buffer = []
                    stream_start = time.time()
                    continue
                
                # Add to write buffer
                write_buffer_ids.extend(batch_ids)
                write_buffer_embeddings.append(embeddings)
                write_buffer_chunks.extend(chunks)
                
                embedded_ids.update(batch_ids)
                total_new += len(batch_ids)
                chunks_buffer = []
                
                # Update stats
                elapsed = time.time() - start_time
                current_rate = total_new / elapsed if elapsed > 0 else 0
                
                ingestion_stats.update({
                    "total_processed": len(embedded_ids),
                    "total_embedded": total_new,
                    "faiss_added": total_faiss_success,
                    "opensearch_indexed": total_opensearch_success,
                    "current_rate": round(current_rate, 2),
                    "last_update": time.time()
                })
                
                # Log progress with timing breakdown
                if total_new % 1000 == 0:
                    batches = total_new / BATCH_SIZE
                    avg_stream = (total_stream_time / batches) * 1000 if batches > 0 else 0
                    avg_embed = (total_embed_time / batches) * 1000 if batches > 0 else 0
                    
                    logger.info(
                        f"✓ {total_new:,} chunks | {current_rate:.1f}/sec | "
                        f"Stream: {avg_stream:.0f}ms | Embed: {avg_embed:.0f}ms | "
                        f"Buffer: {len(write_buffer_ids):,}/{WRITE_BUFFER_SIZE:,}"
                    )
                
                # Write to FAISS/OpenSearch when buffer is full
                if len(write_buffer_ids) >= WRITE_BUFFER_SIZE:
                    combined_embeddings = np.vstack(write_buffer_embeddings)
                    
                    # Send to FAISS (TIMED)
                    faiss_start = time.time()
                    faiss_success = send_to_faiss_bulk(write_buffer_ids, combined_embeddings)
                    faiss_time = time.time() - faiss_start
                    total_faiss_time += faiss_time
                    
                    if faiss_success:
                        total_faiss_success += len(write_buffer_ids)
                    
                    # Send to OpenSearch (TIMED)
                    opensearch_start = time.time()
                    opensearch_success = send_to_opensearch_bulk(write_buffer_ids, write_buffer_chunks)
                    opensearch_time = time.time() - opensearch_start
                    total_opensearch_time += opensearch_time
                    
                    if opensearch_success:
                        total_opensearch_success += len(write_buffer_ids)
                    
                    save_embedded_ids(embedded_ids)
                    logger.info(f"💾 Saved state | FAISS: {faiss_time:.1f}s | OpenSearch: {opensearch_time:.1f}s")
                    
                    
                    write_buffer_ids = []
                    write_buffer_embeddings = []
                    write_buffer_chunks = []
                    
                    del combined_embeddings
                    gc.collect()
                
                del embeddings
                gc.collect()
                
                # Restart streaming timer
                stream_start = time.time()
        
        # Process remaining chunks
        if chunks_buffer:
            logger.info(f"Processing final {len(chunks_buffer)} chunks...")
            texts, ids, chunks = zip(*chunks_buffer)
            
            batch_ids, embeddings = embed_batch(list(texts), list(ids))
            
            if len(batch_ids) > 0:
                write_buffer_ids.extend(batch_ids)
                write_buffer_embeddings.append(embeddings)
                write_buffer_chunks.extend(chunks)
                embedded_ids.update(batch_ids)
                total_new += len(batch_ids)
        
        # Write remaining buffer
        if write_buffer_ids:
            logger.info(f"Writing final {len(write_buffer_ids):,} chunks...")
            combined_embeddings = np.vstack(write_buffer_embeddings)
            
            send_to_faiss_bulk(write_buffer_ids, combined_embeddings)
            send_to_opensearch_bulk(write_buffer_ids, write_buffer_chunks)
        
        save_embedded_ids(embedded_ids)
        
        # Final stats with timing breakdown
        elapsed = time.time() - start_time
        avg_rate = total_new / elapsed if elapsed > 0 else 0
        
        stream_pct = (total_stream_time / elapsed * 100) if elapsed > 0 else 0
        embed_pct = (total_embed_time / elapsed * 100) if elapsed > 0 else 0
        faiss_pct = (total_faiss_time / elapsed * 100) if elapsed > 0 else 0
        opensearch_pct = (total_opensearch_time / elapsed * 100) if elapsed > 0 else 0
        
        ingestion_stats.update({
            "status": "completed",
            "total_processed": len(embedded_ids),
            "total_embedded": total_new,
            "faiss_added": total_faiss_success,
            "opensearch_indexed": total_opensearch_success,
            "current_rate": round(avg_rate, 2),
            "last_update": time.time()
        })
        
        logger.info("="*60)
        logger.info("INGESTION COMPLETE - TIMING BREAKDOWN")
        logger.info(f"New chunks: {total_new:,}")
        logger.info(f"Total processed: {len(embedded_ids):,}")
        logger.info(f"Time: {elapsed:.1f}s | Rate: {avg_rate:.1f} chunks/sec")
        logger.info(f"")
        logger.info(f"Time breakdown:")
        logger.info(f"  Streaming:   {total_stream_time:.1f}s ({stream_pct:.1f}%) ⚠️ KEY METRIC")
        logger.info(f"  Embedding:   {total_embed_time:.1f}s ({embed_pct:.1f}%)")
        logger.info(f"  FAISS:       {total_faiss_time:.1f}s ({faiss_pct:.1f}%)")
        logger.info(f"  OpenSearch:  {total_opensearch_time:.1f}s ({opensearch_pct:.1f}%)")
        logger.info(f"")
        logger.info(f"Chunks streamed from blob: {chunks_streamed:,}")
        logger.info(f"FAISS: {total_faiss_success:,} | OpenSearch: {total_opensearch_success:,}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Ingestion error: {e}", exc_info=True)
        ingestion_stats["status"] = "error"
        ingestion_stats["error"] = str(e)
        
    finally:
        is_ingesting = False
        save_embedded_ids(embedded_ids)

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Wikipedia Embedding Service",
    description="Incremental embedding ingestion service for Wikipedia chunks",
    version="1.0.0"
)

# =========================================================
# RESPONSE MODELS
# =========================================================
class HealthResponse(BaseModel):
    status: str
    embedded_chunks: int
    model_loaded: bool
    is_ingesting: bool

class IngestResponse(BaseModel):
    message: str
    status: str

class StatusResponse(BaseModel):
    status: str
    total_processed: int
    total_embedded: int
    faiss_added: int
    opensearch_indexed: int
    current_rate: float
    is_ingesting: bool
    error: str | None

# =========================================================
# API ENDPOINTS
# =========================================================
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Wikipedia Embedding Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "ingest": "POST /ingest",
            "status": "GET /status"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {
        "status": "ok",
        "embedded_chunks": len(embedded_ids),
        "model_loaded": model is not None,
        "is_ingesting": is_ingesting
    }

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def start_ingestion(background_tasks: BackgroundTasks):
    global is_ingesting
    
    if is_ingesting:
        raise HTTPException(
            status_code=409,
            detail="Ingestion already in progress. Check /status for details."
        )
    
    background_tasks.add_task(run_ingestion)
    logger.info("Ingestion started via API")
    
    return {
        "message": "Ingestion started successfully",
        "status": "running"
    }

@app.get("/status", response_model=StatusResponse, tags=["Ingestion"])
def get_status():
    return {
        "status": ingestion_stats["status"],
        "total_processed": len(embedded_ids),
        "total_embedded": ingestion_stats["total_embedded"],
        "faiss_added": ingestion_stats["faiss_added"],
        "opensearch_indexed": ingestion_stats["opensearch_indexed"],
        "current_rate": ingestion_stats["current_rate"],
        "is_ingesting": is_ingesting,
        "error": ingestion_stats.get("error")
    }

# =========================================================
# STARTUP EVENT
# =========================================================
@app.on_event("startup")
async def startup_event():
    logger.info("="*60)
    logger.info("Wikipedia Embedding Service Started")
    logger.info(f"Model: sentence-transformers/all-MiniLM-L6-v2 (FastEmbed)")
    logger.info(f"Embedding dimension: {EMBED_DIM}")
    logger.info(f"Embedding batch size: {BATCH_SIZE}")
    logger.info(f"Write buffer size: {WRITE_BUFFER_SIZE:,}")
    logger.info(f"Previously embedded: {len(embedded_ids):,} chunks")
    logger.info(f"FAISS endpoint: {FAISS_ENDPOINT}/add")
    logger.info(f"OpenSearch endpoint: {OPENSEARCH_ENDPOINT}/add")
    logger.info("="*60)

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


