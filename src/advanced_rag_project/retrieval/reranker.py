from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder

from advanced_rag_project.documents.models import Chunk


@dataclass
class RerankedResult:
    chunk: Chunk
    rerank_score: float
    hybrid_score: float
    dense_score: float
    keyword_score: float
    document_id: str
    chunk_id: int
    source: str


class CrossEncoderReranker:
    """
    Reranks retrieved candidates using a cross-encoder.

    Unlike embedding models, a cross-encoder receives the
    query and document text together and directly predicts
    their relevance.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 8,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        print(f"Loading reranker model: {model_name}")
        print(f"Reranker device: {device}")

        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(
        self,
        query: str,
        candidates: list,
        top_k: int = 3,
    ) -> list[RerankedResult]:

        if not query or not query.strip():
            raise ValueError("Query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not candidates:
            return []

        pairs = [
            [query, candidate.chunk.text]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        results = []

        for candidate, score in zip(candidates, scores):
            results.append(
                RerankedResult(
                    chunk=candidate.chunk,
                    rerank_score=float(score),
                    hybrid_score=float(candidate.hybrid_score),
                    dense_score=float(candidate.dense_score),
                    keyword_score=float(candidate.keyword_score),
                    document_id=candidate.document_id,
                    chunk_id=candidate.chunk_id,
                    source=candidate.source,
                )
            )

        results.sort(
            key=lambda result: result.rerank_score,
            reverse=True,
        )

        return results[:top_k]