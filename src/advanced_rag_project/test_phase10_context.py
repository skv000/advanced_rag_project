from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.rag.context_engineering import (
    ContextEngineer,
)


def create_test_results():
    """
    Create controlled retrieval results.

    We intentionally include:

    - high relevance chunks
    - duplicate content
    - low relevance content
    - enough content to exercise the context budget
    """

    chunk_1 = Chunk(
        text=(
            "RAG retrieves relevant documents before "
            "sending context to a language model."
        ),
        chunk_id=0,
        source="rag.txt",
    )

    chunk_2 = Chunk(
        text=(
            "Dense retrieval uses embeddings to find "
            "semantically similar chunks."
        ),
        chunk_id=1,
        source="rag.txt",
    )

    chunk_3 = Chunk(
        text=(
            "RAG retrieves relevant documents before "
            "sending context to a language model."
        ),
        chunk_id=2,
        source="another_rag.txt",
    )

    chunk_4 = Chunk(
        text=(
            "The Bhagavad Gita discusses knowledge, "
            "action, and devotion."
        ),
        chunk_id=0,
        source="philosophy.txt",
    )

    return [
        type(
            "Result",
            (),
            {
                "chunk": chunk_1,
                "rerank_score": 0.95,
                "document_id": "rag",
                "source": "rag.txt",
                "chunk_id": 0,
            },
        )(),
        type(
            "Result",
            (),
            {
                "chunk": chunk_2,
                "rerank_score": 0.80,
                "document_id": "rag",
                "source": "rag.txt",
                "chunk_id": 1,
            },
        )(),
        type(
            "Result",
            (),
            {
                "chunk": chunk_3,
                "rerank_score": 0.75,
                "document_id": "another_rag",
                "source": "another_rag.txt",
                "chunk_id": 2,
            },
        )(),
        type(
            "Result",
            (),
            {
                "chunk": chunk_4,
                "rerank_score": 0.20,
                "document_id": "philosophy",
                "source": "philosophy.txt",
                "chunk_id": 0,
            },
        )(),
    ]


def main():

    print("=" * 60)
    print("PHASE 10 — CONTEXT ENGINEERING TEST")
    print("=" * 60)

    results = create_test_results()

    # ---------------------------------------------------------
    # 1. Basic context engineering
    # ---------------------------------------------------------

    engineer = ContextEngineer(
        max_context_tokens=500,
    )

    context_result = engineer.build(
        results
    )

    print("\nFinal context:")
    print("-" * 60)
    print(context_result.context)

    print("\nContext diagnostics:")
    print(
        f"Selected chunks       : "
        f"{len(context_result.items)}"
    )

    print(
        f"Estimated tokens      : "
        f"{context_result.estimated_tokens}"
    )

    print(
        f"Dropped duplicates    : "
        f"{context_result.dropped_duplicates}"
    )

    print(
        f"Dropped low relevance : "
        f"{context_result.dropped_low_relevance}"
    )

    print(
        f"Dropped by budget     : "
        f"{context_result.dropped_budget}"
    )

    # ---------------------------------------------------------
    # 2. Duplicate detection
    # ---------------------------------------------------------

    assert context_result.dropped_duplicates == 1

    # ---------------------------------------------------------
    # 3. Relevance ordering
    # ---------------------------------------------------------

    scores = [
        item.score
        for item in context_result.items
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # ---------------------------------------------------------
    # 4. Duplicate content should appear only once
    # ---------------------------------------------------------

    occurrences = context_result.context.count(
        "RAG retrieves relevant documents"
    )

    assert occurrences == 1

    # ---------------------------------------------------------
    # 5. Token budget
    # ---------------------------------------------------------

    assert (
        context_result.estimated_tokens
        <= 500
    )

    # ---------------------------------------------------------
    # 6. Minimum relevance filtering
    # ---------------------------------------------------------

    filtered_engineer = ContextEngineer(
        max_context_tokens=500,
        min_score=0.5,
    )

    filtered_result = filtered_engineer.build(
        results
    )

    assert (
        filtered_result.dropped_low_relevance
        == 1
    )

    assert all(
        item.score >= 0.5
        for item in filtered_result.items
    )

    # ---------------------------------------------------------
    # 7. Empty results
    # ---------------------------------------------------------

    empty_result = engineer.build([])

    assert empty_result.context == ""
    assert empty_result.items == []
    assert empty_result.estimated_tokens == 0

    # ---------------------------------------------------------
    # 8. Validation
    # ---------------------------------------------------------

    try:
        ContextEngineer(
            max_context_tokens=0
        )

        raise AssertionError(
            "Zero token budget should fail"
        )

    except ValueError:
        pass

    try:
        ContextEngineer(
            max_context_tokens=100,
            min_score=-1,
        )

        raise AssertionError(
            "Negative min_score should fail"
        )

    except ValueError:
        pass

    print("\n" + "=" * 60)
    print("PHASE 10 CONTEXT ENGINEERING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()