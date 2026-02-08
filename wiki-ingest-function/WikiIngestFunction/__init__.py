"""
Memory-Efficient Streaming Azure Function for Chunking Wiki Documents
Handles 100MB+ blobs without OutOfMemoryException
Streams JSON -> Creates chunks -> Stores metadata + chunks
"""

import logging
import os
import json
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient
import azure.functions as func
import requests


# Config
PROCESSED_CONTAINER = "wiki-processed"
FAILED_CONTAINER = "wiki-failed"
METADATA_CONTAINER = "wiki-metadata"
CHUNKS_BLOB = "documents_chunks.jsonl"
METADATA_BLOB = "documents_metadata.jsonl"
INGESTED_DOCS_BLOB = "ingested_docs.txt"
PROGRESS_INTERVAL = 10_000
BATCH_APPEND_SIZE = 50_000
STREAMING_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
EMBEDDING_CHUNK_SIZE = 1000  # characters per embedding chunk

logger = logging.getLogger(__name__)


def get_blob_service():
    conn_str = os.environ.get("WikiStorageConnection")
    if not conn_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment")
    return BlobServiceClient.from_connection_string(conn_str)


def ensure_append_blob_exists(container_client, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    try:
        blob_client.get_blob_properties()
    except Exception:
        try:
            blob_client.create_append_blob()
            logger.info(f"Created append blob: {blob_name}")
        except Exception as e:
            logger.warning(f"Could not create append blob {blob_name}: {e}")
    return blob_client


def load_processed_doc_ids(ingested_docs_blob):
    processed_doc_ids = set()
    try:
        downloader = ingested_docs_blob.download_blob()
        for chunk in downloader.chunks():
            lines = chunk.decode("utf-8").splitlines()
            processed_doc_ids.update(line.strip() for line in lines if line.strip())
        logger.info(f"Loaded {len(processed_doc_ids):,} previously ingested documents")
    except Exception as e:
        logger.info(f"No previous ingestion data found: {e}")
    return processed_doc_ids


def stream_parse_json_array(blob_stream):
    buffer = ""
    in_array = False
    bracket_depth = 0
    current_object = ""
    downloader = blob_stream.download_blob()
    
    for chunk in downloader.chunks():
        buffer += chunk.decode('utf-8')
        i = 0
        while i < len(buffer):
            char = buffer[i]
            if char == '[' and not in_array:
                in_array = True
                i += 1
                continue
            if not in_array:
                i += 1
                continue
            if char == '{':
                bracket_depth += 1
                current_object += char
            elif char == '}':
                current_object += char
                bracket_depth -= 1
                if bracket_depth == 0:
                    try:
                        obj = json.loads(current_object.strip())
                        yield obj
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse object: {e}")
                    current_object = ""
            elif bracket_depth > 0:
                current_object += char
            i += 1
        buffer = current_object if bracket_depth > 0 else ""


def chunk_text(text, chunk_size=EMBEDDING_CHUNK_SIZE):
    """Split text into fixed-size chunks for embedding"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size
    return chunks

def trigger_embedding_service():
    embedding_api = os.environ.get("EmbeddingApi")
    if not embedding_api:
        logger.warning("EmbeddingApi not configured, skipping embedding trigger")
        return

    try:
        resp = requests.post(embedding_api, timeout=10)
        if resp.status_code == 200:
            logger.info("🚀 Successfully triggered embedding service")
        else:
            logger.warning(
                f"Embedding service responded with {resp.status_code}: {resp.text}"
            )
    except Exception as e:
        logger.error(f"Failed to trigger embedding service: {e}")



def main(myblob: func.InputStream):
    blob_name = myblob.name.split('/')[-1]
    logger.info(f"Function triggered by blob: {blob_name}, size: {myblob.length / 1024 / 1024:.2f} MB")
    
    try:
        blob_service = get_blob_service()
        processed_client = blob_service.get_container_client(PROCESSED_CONTAINER)
        failed_client = blob_service.get_container_client(FAILED_CONTAINER)
        metadata_client = blob_service.get_container_client(METADATA_CONTAINER)
        
        source_container = myblob.name.split('/')[0] if '/' in myblob.name else "wiki-docs"
        source_blob_client = blob_service.get_blob_client(source_container, blob_name)
        
        # Setup append blobs
        chunks_blob = ensure_append_blob_exists(metadata_client, CHUNKS_BLOB)
        metadata_blob = ensure_append_blob_exists(metadata_client, METADATA_BLOB)
        ingested_docs_blob = ensure_append_blob_exists(metadata_client, INGESTED_DOCS_BLOB)
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure clients: {e}")
        raise
    
    processed_doc_ids = load_processed_doc_ids(ingested_docs_blob)
    
    scanned = 0
    skipped = 0
    ingested = 0
    new_metadata_records = []
    new_doc_ids = []
    new_chunk_records = []
    
    try:
        for article in stream_parse_json_array(source_blob_client):
            scanned += 1
            doc_id = str(article.get("id", ""))
            if not doc_id:
                skipped += 1
                continue
            if doc_id in processed_doc_ids:
                skipped += 1
                continue
            
            title = article.get("title", "")
            text = article.get("text", "")
            
            # Chunk the text
            chunks = chunk_text(text)
            for idx, chunk_text_content in enumerate(chunks):
                chunk_record = {
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text_content,
                    "source_blob": blob_name,
                    "status": "pending_embedding",
                    "ingested_at": datetime.now(timezone.utc).isoformat()
                }
                new_chunk_records.append(json.dumps(chunk_record))
            
            # Metadata record
            record = {
                "doc_id": doc_id,
                "title": title,
                "source_blob": blob_name,
                "num_chunks": len(chunks),
                "status": "ingested",
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
            new_metadata_records.append(json.dumps(record))
            new_doc_ids.append(doc_id)
            processed_doc_ids.add(doc_id)
            ingested += 1
            
            # Flush periodically
            if len(new_metadata_records) >= BATCH_APPEND_SIZE:
                # Metadata
                metadata_blob.append_block("\n".join(new_metadata_records) + "\n")
                new_metadata_records.clear()
                # Chunks
                chunks_blob.append_block("\n".join(new_chunk_records) + "\n")
                new_chunk_records.clear()
                # Doc IDs
                ingested_docs_blob.append_block("\n".join(new_doc_ids) + "\n")
                new_doc_ids.clear()
        
        # Flush remaining
        if new_metadata_records:
            metadata_blob.append_block("\n".join(new_metadata_records) + "\n")
        if new_chunk_records:
            chunks_blob.append_block("\n".join(new_chunk_records) + "\n")
        if new_doc_ids:
            ingested_docs_blob.append_block("\n".join(new_doc_ids) + "\n")
        
        # Archive raw JSON
        dest_blob = processed_client.get_blob_client(blob_name)
        dest_blob.start_copy_from_url(source_blob_client.url)
        import time
        elapsed = 0
        max_wait = 300
        while elapsed < max_wait:
            props = dest_blob.get_blob_properties()
            if props.copy.status == 'success':
                break
            elif props.copy.status == 'failed':
                raise Exception(f"Copy failed: {props.copy.status_description}")
            time.sleep(2)
            elapsed += 2
        source_blob_client.delete_blob()
        
        logger.info(f"✅ Completed processing {blob_name}: scanned={scanned}, ingested={ingested}, skipped={skipped}")

        trigger_embedding_service()
    
    except Exception as e:
        logger.error(f"Error processing blob {blob_name}: {e}")
        # Move to failed container
        try:
            dest_blob = failed_client.get_blob_client(blob_name)
            dest_blob.start_copy_from_url(source_blob_client.url)
            source_blob_client.delete_blob()
        except Exception as move_error:
            logger.error(f"Could not move blob to failed container: {move_error}")
        raise
