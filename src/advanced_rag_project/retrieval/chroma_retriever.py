from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


class ChromaRetriever:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_documents",
    ):
        self.embedder = Embedder()

        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:

        query_embedding = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        chunks = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            chunk = Chunk(
                text=document,
                chunk_id=int(metadata["chunk_id"]),
                source=metadata["source"],
            )

            similarity = 1 / (1 + distance)

            chunks.append(
                (chunk, similarity)
            )

        return chunks