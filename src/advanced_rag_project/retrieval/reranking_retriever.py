from advanced_rag_project.retrieval.hybrid_retriever import HybridRetriever
from advanced_rag_project.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedResult,
)


class RerankingRetriever:
    """
    Retrieval pipeline:

        Dense + Keyword Retrieval
                    ↓
                Candidates
                    ↓
             Cross-Encoder
                    ↓
              Final Results
    """

    def __init__(
        self,
        chunks,
        persist_directory: str = "data/chroma",
        collection_name: str = "documents",
        reranker_model: str = CrossEncoderReranker.DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 8,
    ):
        self.hybrid_retriever = HybridRetriever(
            chunks=chunks,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        self.reranker = CrossEncoderReranker(
            model_name=reranker_model,
            device=device,
            batch_size=batch_size,
        )

    def search(
        self,
        query: str,
        candidate_k: int = 10,
        top_k: int = 3,
        dense_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[RerankedResult]:

        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if top_k > candidate_k:
            raise ValueError(
                "top_k cannot be greater than candidate_k"
            )

        candidates = self.hybrid_retriever.search(
            query=query,
            top_k=candidate_k,
            dense_weight=dense_weight,
            keyword_weight=keyword_weight,
        )

        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )