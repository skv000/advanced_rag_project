from advanced_rag_project.rag.context_engineering import (
    ContextEngineer,
    ContextResult,
)


def build_context_result(
    results: list,
    max_context_tokens: int = 1500,
    min_score: float | None = None,
) -> ContextResult:
    """
    Build structured context from retrieval results.
    """

    engineer = ContextEngineer(
        max_context_tokens=max_context_tokens,
        min_score=min_score,
    )

    return engineer.build(results)


def build_context(
    results: list,
    max_context_tokens: int = 1500,
    min_score: float | None = None,
) -> str:
    """
    Backward-compatible helper returning only
    the final context string.
    """

    result = build_context_result(
        results=results,
        max_context_tokens=max_context_tokens,
        min_score=min_score,
    )

    return result.context