"""FastAPI backend for KnowledgeOps AI.

Run:
    uvicorn backend.main:app --reload
Docs:
    http://localhost:8000/docs
"""

import os
import time
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


from backend.dependencies import (
    get_embedder,
    get_llm,
    get_provider_name,
    get_store,
)
from backend.models import (
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    SourceModel,
)
from rag.answer import answer_question
from rag.chunker import chunk_document
from rag.incremental import run_incremental
from rag.pipeline import index_document

import time
import logging

from fastapi import Request
from backend.logging_config import (
    configure_logging,
    new_request_id,
    request_id_var,
)

app = FastAPI(
    title="KnowledgeOps AI",
    description="RAG knowledge assistant with citations",
    version="0.1.0",
)

# Allow the Streamlit UI (Milestone 6) to call this API from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure structured logging at import time
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("knowledgeops.api")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Assign a request ID, time the request, log the outcome."""
    req_id = new_request_id()
    request_id_var.set(req_id)
    start = time.perf_counter()

    try:
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            "request completed",
            extra={
                "event": "http_request",
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        response.headers["X-Request-ID"] = req_id
        return response
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.exception(
            "request failed",
            extra={
                "event": "http_error",
                "method": request.method,
                "path": request.url.path,
                "status": 500,
                "latency_ms": latency_ms,
            },
        )
        raise

# ============================================================
# Health check
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_store()
    return HealthResponse(
        status="ok",
        vector_store_count=store.count(),
        llm_provider=get_provider_name(),
    )

@app.get("/live")
def live() -> dict:
    """Liveness probe: process is up."""
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict:
    """Readiness probe: dependencies are reachable."""
    try:
        store = get_store()
        count = store.count()
        return {
            "status": "ready",
            "vector_store_count": count,
            "llm_provider": get_provider_name(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")
        
# ============================================================
# /query — ask a question
# ============================================================

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = answer_question(
            request.question,
            top_k=request.top_k,
            embedder=get_embedder(),
            store=get_store(),
            llm_client=get_llm(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        found=result.found,
        sources=[
            SourceModel(
                index=s.index,
                source_name=s.source_name,
                page_number=s.page_number,
                text=s.text,
            )
            for s in result.sources
        ],
    )


# ============================================================
# /documents — list, upload, index
# ============================================================

PROCESSED_DIR = Path("data/processed")


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """
    List documents currently available as processed JSON.

    Note: This lists files in data/processed/. It does not verify that
    they're indexed in the vector store.
    """
    import json

    if not PROCESSED_DIR.exists():
        return DocumentListResponse(documents=[], total_chunks=0)

    docs: list[DocumentInfo] = []
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Count chunks produced from this doc
            chunks = chunk_document(data)
            docs.append(
                DocumentInfo(
                    source_name=data["source_name"],
                    total_pages=data["total_pages"],
                    chunk_count=len(chunks),
                )
            )
        except Exception:
            continue

    return DocumentListResponse(
        documents=docs,
        total_chunks=get_store().count(),
    )


@app.post("/documents/index", response_model=IndexResponse)
def reindex_all() -> IndexResponse:
    """
    Re-index every JSON file in data/processed/.

    Wipes the vector store and rebuilds it from scratch.
    """
    store = get_store()
    embedder = get_embedder()

    store.reset()

    indexed_files = 0
    total_chunks = 0
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        n = index_document(path, embedder, store)
        indexed_files += 1
        total_chunks += n

    if indexed_files == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found in {PROCESSED_DIR}. Run ingestion first.",
        )

        from rag.incremental import run_incremental


@app.post("/documents/incremental", response_model=IndexResponse)
def reindex_incremental() -> IndexResponse:
    """
    Reindex only documents whose content hash has changed.

    Returns a summary of what was processed vs. skipped.
    """
    result = run_incremental(force=False, dry_run=False)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return IndexResponse(
        source_name=f"{len(result['new_or_changed'])} changed of {result['total_pdfs']}",
        chunks_indexed=result["indexed_chunks"],
        message=(
            f"Processed {len(result['new_or_changed'])} file(s), "
            f"skipped {len(result['unchanged'])}. "
            f"Deleted {result['deleted_chunks']} stale chunks, "
            f"added {result['indexed_chunks']} new chunks."
        ),
    )

    return IndexResponse(
        source_name=f"{indexed_files} document(s)",
        chunks_indexed=total_chunks,
        message=f"Re-indexed {indexed_files} file(s), {total_chunks} chunks total.",
    )


if __name__ == "__main__":
    # Allow `python -m backend.main` as an alternative to uvicorn
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)