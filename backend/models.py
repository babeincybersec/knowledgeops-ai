"""Pydantic models for API request/response."""

from pydantic import BaseModel, Field


# ---------- /query ----------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          description="The question to ask")
    top_k: int = Field(5, ge=1, le=20,
                       description="Number of chunks to retrieve")


class SourceModel(BaseModel):
    index: int
    source_name: str
    page_number: int
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[SourceModel]


# ---------- /documents ----------

class DocumentInfo(BaseModel):
    source_name: str
    total_pages: int
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class IndexResponse(BaseModel):
    source_name: str
    chunks_indexed: int
    message: str


# ---------- /health ----------

class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    llm_provider: str