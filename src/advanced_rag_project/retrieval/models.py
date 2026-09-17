from dataclasses import dataclass

from advanced_rag_project.documents.models import Chunk


@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result.
    """

    chunk: Chunk
    similarity: float
    document_id: str
    chunk_id: int
    source: str