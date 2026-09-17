from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.retrieval.models import RetrievalResult
from advanced_rag_project.vectorstore.chroma_store import (
    ChromaVectorStore,
)


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
        similarity_threshold: float | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Search ChromaDB and return ranked retrieval results.

        Parameters
        ----------
        query:
            User search query.

        top_k:
            Maximum number of results to return.

        similarity_threshold:
            Optional minimum similarity score.

        filters:
            Optional Chroma metadata filters.

        Returns
        -------
        list[RetrievalResult]
            Ranked retrieval results.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if (
            similarity_threshold is not None
            and not 0 <= similarity_threshold <= 1
        ):
            raise ValueError(
                "similarity_threshold must be "
                "between 0 and 1."
            )

        query_embedding = self.embedder.embed_text(
            query
        )

        # Retrieve more candidates when filtering
        # by similarity.
        candidate_k = top_k

        if similarity_threshold is not None:
            candidate_k = max(
                top_k * 3,
                top_k,
            )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            filters=filters,
        )

        retrieval_results = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for (
            chunk_id,
            document,
            metadata,
            distance,
        ) in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            similarity = 1 / (1 + distance)

            if (
                similarity_threshold is not None
                and similarity < similarity_threshold
            ):
                continue

            chunk = Chunk(
                text=document,
                chunk_id=int(
                    metadata["chunk_id"]
                ),
                source=metadata["source"],
            )

            result = RetrievalResult(
                chunk=chunk,
                similarity=similarity,
                document_id=metadata[
                    "document_id"
                ],
                chunk_id=int(
                    metadata["chunk_id"]
                ),
                source=metadata["source"],
            )

            retrieval_results.append(result)

        retrieval_results.sort(
            key=lambda result: result.similarity,
            reverse=True,
        )

        return retrieval_results[:top_k]