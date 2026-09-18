from advanced_rag_project.rag.context_engineering import (
    ContextEngineer,
)


def build_context(
    results: list,
    max_context_tokens: int = 1500,
    min_score: float | None = None,
) -> str:
    """
    Build the final RAG context using the context engineering layer.

    This function remains as a simple compatibility wrapper around
    ContextEngineer.
    """

    engineer = ContextEngineer(
        max_context_tokens=max_context_tokens,
        min_score=min_score,
    )

    result = engineer.build(
        results
    )

    return result.context