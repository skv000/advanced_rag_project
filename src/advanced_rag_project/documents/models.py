from dataclasses import dataclass


@dataclass
class Chunk:
    """
    Represents a chunk of a source document.
    """

    text: str
    chunk_id: int
    source: str


@dataclass
class Document:
    """
    Represents a loaded source document.
    """

    text: str
    source: str
    document_id: str
    content_hash: str

    file_type: str
    file_size: int
    modified_time: float