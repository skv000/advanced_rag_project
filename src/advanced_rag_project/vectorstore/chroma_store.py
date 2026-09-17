import chromadb

from advanced_rag_project.documents.models import Chunk


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_documents",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        self.collection.add(
            ids=[str(chunk.chunk_id) for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )