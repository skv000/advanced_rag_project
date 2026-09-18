from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.retrieval.chroma_retriever import (
    ChromaRetriever,
)
from advanced_rag_project.retrieval.keyword_retriever import (
    KeywordRetriever,
)
from advanced_rag_project.retrieval.models import (
    RetrievalResult,
)


class HybridRetrievalResult:
    """
    Combined dense + keyword retrieval result.
    """

    def __init__(
        self,
        chunk: Chunk,
        dense_score: float,
        keyword_score: float,
        hybrid_score: float,
        document_id: str,
        source: str,
        chunk_id: int,
    ):
        self.chunk = chunk
        self.dense_score = dense_score
        self.keyword_score = keyword_score
        self.hybrid_score = hybrid_score
        self.document_id = document_id
        self.source = source
        self.chunk_id = chunk_id

    def __repr__(self):
        return (
            "HybridRetrievalResult("
            f"hybrid={self.hybrid_score:.4f}, "
            f"dense={self.dense_score:.4f}, "
            f"keyword={self.keyword_score:.4f}, "
            f"source={self.source!r}"
            ")"
        )


class HybridRetriever:
    """
    Combines dense vector retrieval and keyword retrieval.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_documents",
    ):

        self.keyword_retriever = (
            KeywordRetriever(chunks)
        )

        self.dense_retriever = ChromaRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    @staticmethod
    def _normalize_scores(
        results: list[RetrievalResult],
    ) -> dict[str, float]:
        """
        Min-max normalize scores to 0-1.
        """

        if not results:
            return {}

        scores = [
            result.similarity
            for result in results
        ]

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:

            return {
                result.source: 1.0
                for result in results
            }

        return {
            result.source: (
                (result.similarity - minimum)
                / (maximum - minimum)
            )
            for result in results
        }

    def search(
        self,
        query: str,
        top_k: int = 3,
        dense_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[HybridRetrievalResult]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if dense_weight < 0:
            raise ValueError(
                "dense_weight cannot be negative."
            )

        if keyword_weight < 0:
            raise ValueError(
                "keyword_weight cannot be negative."
            )

        total_weight = (
            dense_weight
            + keyword_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one retrieval weight "
                "must be greater than zero."
            )

        # Normalize weights
        dense_weight /= total_weight
        keyword_weight /= total_weight

        candidate_k = max(
            top_k * 3,
            top_k,
        )

        dense_results = (
            self.dense_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        keyword_results = (
            self.keyword_retriever.search(
                query=query,
                top_k=candidate_k,
            )
        )

        dense_scores = (
            self._normalize_scores(
                dense_results
            )
        )

        keyword_scores = (
            self._normalize_scores(
                keyword_results
            )
        )

        # Map results by source + chunk ID.
        combined = {}

        for result in dense_results:

            key = (
                result.source,
                result.chunk_id,
            )

            combined[key] = {
                "chunk": result.chunk,
                "document_id": result.document_id,
                "source": result.source,
                "chunk_id": result.chunk_id,
                "dense_score": dense_scores.get(
                    result.source,
                    0.0,
                ),
                "keyword_score": 0.0,
            }

        for result in keyword_results:

            key = (
                result.source,
                result.chunk_id,
            )

            if key not in combined:

                combined[key] = {
                    "chunk": result.chunk,
                    "document_id": result.document_id,
                    "source": result.source,
                    "chunk_id": result.chunk_id,
                    "dense_score": 0.0,
                    "keyword_score": 0.0,
                }

            combined[key][
                "keyword_score"
            ] = keyword_scores.get(
                result.source,
                0.0,
            )

        results = []

        for item in combined.values():

            hybrid_score = (
                dense_weight
                * item["dense_score"]
                +
                keyword_weight
                * item["keyword_score"]
            )

            results.append(
                HybridRetrievalResult(
                    chunk=item["chunk"],
                    dense_score=item[
                        "dense_score"
                    ],
                    keyword_score=item[
                        "keyword_score"
                    ],
                    hybrid_score=hybrid_score,
                    document_id=item[
                        "document_id"
                    ],
                    source=item["source"],
                    chunk_id=item["chunk_id"],
                )
            )

        results.sort(
            key=lambda result: result.hybrid_score,
            reverse=True,
        )

        return results[:top_k]