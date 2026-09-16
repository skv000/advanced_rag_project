from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.retrieval.similarity import cosine_similarity


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embedder = Embedder()

        print("Creating chunk embeddings...")

        self.embeddings = self.embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:

        query_embedding = self.embedder.embed_text(query)

        results = []

        for chunk, embedding in zip(
            self.chunks,
            self.embeddings,
        ):
            score = cosine_similarity(
                query_embedding,
                embedding,
            )

            results.append(
                (chunk, score)
            )

        results.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return results[:top_k]
