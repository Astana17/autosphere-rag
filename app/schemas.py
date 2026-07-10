from pydantic import BaseModel, Field, conlist
from typing import List, Dict, Any, Optional

class Chunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the text chunk")
    text: str = Field(..., description="The content of the chunk")
    source_file: str = Field(..., description="Filename of the original document")
    page: Optional[int] = Field(None, description="Page number of the original document if applicable")
    section_title: Optional[str] = Field(None, description="Title of the section containing this chunk")
    char_start: Optional[int] = Field(None, description="Starting character index in original document")
    char_end: Optional[int] = Field(None, description="Ending character index in original document")
    document_version: Optional[str] = Field(None, description="Version tag of the document")

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The natural language question to ask the RAG system."
    )
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of source chunks to retrieve.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply to the retrieval search.")

class SourceReference(BaseModel):
    source_file: str
    page: Optional[int] = None
    section_title: Optional[str] = None
    chunk_id: Optional[str] = None
    excerpt: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="The generated response text grounded in retrieved documents.")
    sources: List[SourceReference] = Field(default_factory=list, description="References to the chunks used for grounding.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimated confidence score in range [0, 1].")
    used_context: bool = Field(..., description="True if answer was drawn from context, False if system fell back/refused.")
