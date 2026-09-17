from pathlib import Path
import shutil

from advanced_rag_project.ingestion.pipeline import (
    IngestionPipeline,
)
from advanced_rag_project.retrieval.chroma_retriever import (
    ChromaRetriever,
)


TEST_DIR = Path(
    "data/phase7_retrieval_test"
)

CHROMA_DIR = (
    "data/chroma_phase7_test"
)


def create_documents():

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (TEST_DIR / "python.txt").write_text(
        """
        Python is a programming language.

        Python is widely used for machine learning,
        artificial intelligence, and web development.
        """.strip(),
        encoding="utf-8",
    )

    (TEST_DIR / "rag.md").write_text(
        """
        # Retrieval Augmented Generation

        RAG combines information retrieval with
        language model generation.

        A retriever finds relevant documents before
        the language model generates an answer.
        """.strip(),
        encoding="utf-8",
    )

    (TEST_DIR / "philosophy.txt").write_text(
        """
        Advaita Vedanta is a non-dual philosophical
        tradition within Vedanta.

        It discusses the relationship between Brahman,
        Atman, and ultimate reality.
        """.strip(),
        encoding="utf-8",
    )


def main():

    create_documents()

    # --------------------------------------------------
    # Ingest
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=CHROMA_DIR,
        collection_name="phase7_test",
    )

    print("\n")
    print("=" * 60)
    print("PHASE 7 - INGESTION")
    print("=" * 60)

    stats = pipeline.ingest_directory(
        str(TEST_DIR)
    )

    assert stats.new == 3
    assert stats.failed == 0

    print(
        "\nIngestion successful."
    )

    # --------------------------------------------------
    # Create retriever
    # --------------------------------------------------

    retriever = ChromaRetriever(
        persist_directory=CHROMA_DIR,
        collection_name="phase7_test",
    )

    # --------------------------------------------------
    # Test 1: Basic retrieval
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST 1 - BASIC RETRIEVAL")
    print("=" * 60)

    results = retriever.search(
        query="What is retrieval augmented generation?",
        top_k=3,
    )

    assert len(results) > 0
    assert len(results) <= 3

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{index}. "
            f"similarity={result.similarity:.4f}"
        )

        print(
            f"source={result.source}"
        )

        print(
            f"document_id={result.document_id}"
        )

        print(
            f"chunk_id={result.chunk_id}"
        )

        print(
            f"text={result.chunk.text[:100]}..."
        )

    # Results must be ranked descending.
    scores = [
        result.similarity
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # --------------------------------------------------
    # Test 2: Top-K
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST 2 - TOP-K")
    print("=" * 60)

    results = retriever.search(
        query="machine learning",
        top_k=2,
    )

    assert len(results) <= 2

    print(
        f"Requested top_k=2"
    )

    print(
        f"Returned={len(results)}"
    )

    # --------------------------------------------------
    # Test 3: Similarity threshold
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST 3 - SIMILARITY THRESHOLD")
    print("=" * 60)

    results = retriever.search(
        query="retrieval augmented generation",
        top_k=5,
        similarity_threshold=0.30,
    )

    assert all(
        result.similarity >= 0.30
        for result in results
    )

    print(
        f"Results above threshold: "
        f"{len(results)}"
    )

    for result in results:
        print(
            f"  {result.similarity:.4f} "
            f"- {result.source}"
        )

    # --------------------------------------------------
    # Test 4: Metadata filter
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST 4 - METADATA FILTER")
    print("=" * 60)

    results = retriever.search(
        query="retrieval augmented generation",
        top_k=5,
        filters={
            "file_type": ".md"
        },
    )

    assert len(results) > 0

    assert all(
        result.source.endswith(".md")
        for result in results
    )

    print(
        f"Markdown results: {len(results)}"
    )

    for result in results:
        print(
            f"  {result.similarity:.4f} "
            f"- {result.source}"
        )

    # --------------------------------------------------
    # Test 5: Document ID filter
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST 5 - DOCUMENT FILTER")
    print("=" * 60)

    results = retriever.search(
        query="non dual philosophy",
        top_k=5,
        filters={
            "document_id": "philosophy"
        },
    )

    assert len(results) > 0

    assert all(
        result.document_id == "philosophy"
        for result in results
    )

    print(
        f"Philosophy document results: "
        f"{len(results)}"
    )

    # --------------------------------------------------
    # Test 6: Invalid inputs
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TEST 6 - INPUT VALIDATION")
    print("=" * 60)

    try:
        retriever.search(
            query="",
            top_k=3,
        )
        raise AssertionError(
            "Empty query should fail."
        )
    except ValueError:
        print(
            "Empty query correctly rejected."
        )

    try:
        retriever.search(
            query="test",
            top_k=0,
        )
        raise AssertionError(
            "top_k=0 should fail."
        )
    except ValueError:
        print(
            "Invalid top_k correctly rejected."
        )

    try:
        retriever.search(
            query="test",
            top_k=3,
            similarity_threshold=1.5,
        )
        raise AssertionError(
            "Invalid threshold should fail."
        )
    except ValueError:
        print(
            "Invalid threshold correctly rejected."
        )

    # --------------------------------------------------
    # Complete
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PHASE 7 RETRIEVAL TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()