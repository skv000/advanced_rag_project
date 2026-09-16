from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    chunk_id: int
    source: str
