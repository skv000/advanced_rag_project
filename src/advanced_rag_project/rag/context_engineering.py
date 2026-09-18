from dataclasses import dataclass

from advanced_rag_project.documents.models import Chunk


@dataclass
class ContextItem:
    """
    A chunk selected for the final RAG context.
    """

    chunk: Chunk
    score: float
    source: str
    chunk_id: int
    document_id: str


@dataclass
class ContextResult:
    """
    Final context produced by the context engineering layer.
    """

    items: list[ContextItem]
    context: str
    estimated_tokens: int
    dropped_duplicates: int
    dropped_low_relevance: int
    dropped_budget: int


class ContextEngineer:
    """
    Prepare retrieved results for the final RAG prompt.

    Responsibilities:

    1. Remove duplicate chunks.
    2. Filter low-relevance results.
    3. Preserve relevance ordering.
    4. Apply a context budget.
    5. Assemble the final context.
    """

    def __init__(
        self,
        max_context_tokens: int = 1500,
        min_score: float | None = None,
    ):
        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens must be greater than 0"
            )

        if min_score is not None and min_score < 0:
            raise ValueError(
                "min_score cannot be negative"
            )

        self.max_context_tokens = max_context_tokens
        self.min_score = min_score

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token estimate.

        We intentionally avoid adding a tokenizer dependency
        at this stage.

        A rough English approximation is:

            tokens ≈ characters / 4
        """

        if not text:
            return 0

        return max(
            1,
            (len(text) + 3) // 4,
        )

    @staticmethod
    def _extract_result_data(result):
        """
        Extract common retrieval information.

        Supports:

        - RerankedResult
        - HybridRetrievalResult
        - RetrievalResult
        - legacy (Chunk, score) tuples
        """

        if isinstance(result, tuple):

            chunk, score = result

            return {
                "chunk": chunk,
                "score": float(score),
                "document_id": "",
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
            }

        chunk = result.chunk

        if hasattr(result, "rerank_score"):
            score = float(
                result.rerank_score
            )
        elif hasattr(result, "hybrid_score"):
            score = float(
                result.hybrid_score
            )
        elif hasattr(result, "similarity"):
            score = float(
                result.similarity
            )
        else:
            score = 0.0

        return {
            "chunk": chunk,
            "score": score,
            "document_id": getattr(
                result,
                "document_id",
                "",
            ),
            "source": getattr(
                result,
                "source",
                chunk.source,
            ),
            "chunk_id": getattr(
                result,
                "chunk_id",
                chunk.chunk_id,
            ),
        }

    @staticmethod
    def _deduplicate(
        items: list[ContextItem],
    ) -> tuple[list[ContextItem], int]:
        """
        Remove duplicate chunks based on normalized text.
        """

        seen = set()
        unique_items = []
        duplicates = 0

        for item in items:

            normalized_text = " ".join(
                item.chunk.text.lower().split()
            )

            if normalized_text in seen:
                duplicates += 1
                continue

            seen.add(normalized_text)
            unique_items.append(item)

        return unique_items, duplicates

    def build(
        self,
        results: list,
    ) -> ContextResult:
        """
        Transform retrieval results into final RAG context.
        """

        if not results:
            return ContextResult(
                items=[],
                context="",
                estimated_tokens=0,
                dropped_duplicates=0,
                dropped_low_relevance=0,
                dropped_budget=0,
            )

        # -----------------------------------------------------
        # Extract results
        # -----------------------------------------------------

        items = []

        for result in results:

            data = self._extract_result_data(
                result
            )

            if not data["chunk"].text.strip():
                continue

            items.append(
                ContextItem(
                    chunk=data["chunk"],
                    score=data["score"],
                    source=data["source"],
                    chunk_id=data["chunk_id"],
                    document_id=data["document_id"],
                )
            )

        # -----------------------------------------------------
        # Sort by relevance
        # -----------------------------------------------------

        items.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        # -----------------------------------------------------
        # Relevance filtering
        # -----------------------------------------------------

        dropped_low_relevance = 0

        if self.min_score is not None:

            filtered_items = []

            for item in items:

                if item.score >= self.min_score:
                    filtered_items.append(item)
                else:
                    dropped_low_relevance += 1

            items = filtered_items

        # -----------------------------------------------------
        # Deduplication
        # -----------------------------------------------------

        items, dropped_duplicates = (
            self._deduplicate(items)
        )

        # -----------------------------------------------------
        # Context budget
        # -----------------------------------------------------

        selected_items = []
        estimated_tokens = 0
        dropped_budget = 0

        for item in items:

            formatted_item = (
                f"[Source: {item.source} | "
                f"Chunk ID: {item.chunk_id} | "
                f"Score: {item.score:.4f}]\n"
                f"{item.chunk.text}"
            )

            item_tokens = self._estimate_tokens(
                formatted_item
            )

            if (
                estimated_tokens + item_tokens
                > self.max_context_tokens
            ):
                dropped_budget += 1
                continue

            selected_items.append(item)

            estimated_tokens += item_tokens

        # -----------------------------------------------------
        # Assemble final context
        # -----------------------------------------------------

        context_parts = []

        for item in selected_items:

            context_parts.append(
                f"[Source: {item.source} | "
                f"Chunk ID: {item.chunk_id} | "
                f"Score: {item.score:.4f}]\n"
                f"{item.chunk.text}"
            )

        context = "\n\n".join(
            context_parts
        )

        return ContextResult(
            items=selected_items,
            context=context,
            estimated_tokens=estimated_tokens,
            dropped_duplicates=dropped_duplicates,
            dropped_low_relevance=dropped_low_relevance,
            dropped_budget=dropped_budget,
        )