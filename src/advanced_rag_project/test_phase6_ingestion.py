from pathlib import Path
import shutil

from advanced_rag_project.ingestion.pipeline import (
    IngestionPipeline,
)


TEST_DIR = Path("data/phase6_ingestion_test")
CHROMA_DIR = "data/chroma_phase6_test"


def reset_test_directory():
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (TEST_DIR / "research").mkdir(
        exist_ok=True
    )


def create_test_documents():

    # TXT
    (TEST_DIR / "python.txt").write_text(
        """
        Python is a programming language.

        Python is widely used for artificial intelligence.
        """.strip(),
        encoding="utf-8",
    )

    # Markdown
    (TEST_DIR / "rag.md").write_text(
        """
        # Retrieval Augmented Generation

        RAG combines retrieval with language generation.

        ## Retrieval

        Relevant documents are retrieved before generation.
        """.strip(),
        encoding="utf-8",
    )

    # Nested TXT
    (TEST_DIR / "research" / "ai.txt").write_text(
        """
        Artificial intelligence systems can process
        information and generate useful outputs.
        """.strip(),
        encoding="utf-8",
    )

    # Empty document
    (TEST_DIR / "empty.txt").write_text(
        "",
        encoding="utf-8",
    )

    # Unsupported
    (TEST_DIR / "research" / "paper.pdf").write_bytes(
        b"fake pdf content"
    )


def main():

    reset_test_directory()
    create_test_documents()

    pipeline = IngestionPipeline(
        persist_directory=CHROMA_DIR,
        collection_name="phase6_test",
    )

    print("\n")
    print("=" * 60)
    print("PHASE 6 - FIRST INGESTION")
    print("=" * 60)

    stats = pipeline.ingest_directory(
        str(TEST_DIR)
    )

    print("\n")
    print("FIRST INGESTION RESULTS")
    print("=" * 60)

    print(f"Scanned:    {stats.scanned}")
    print(f"New:        {stats.new}")
    print(f"Failed:     {stats.failed}")

    assert stats.scanned == 4
    assert stats.new == 3
    assert stats.failed == 1

    count = pipeline.vector_store.collection.count()

    print(
        f"\nChroma record count: {count}"
    )

    assert count == 3

    print(
        "\nTXT + Markdown + recursive ingestion passed."
    )

    print("\n")
    print("=" * 60)
    print("PHASE 6 - SECOND INGESTION")
    print("=" * 60)

    stats = pipeline.ingest_directory(
        str(TEST_DIR)
    )

    print("\n")
    print("SECOND INGESTION RESULTS")
    print("=" * 60)

    print(f"Scanned:    {stats.scanned}")
    print(f"Unchanged:  {stats.unchanged}")
    print(f"Failed:     {stats.failed}")

    assert stats.scanned == 4
    assert stats.unchanged == 3
    assert stats.failed == 1

    final_count = (
        pipeline.vector_store.collection.count()
    )

    print(
        f"\nFinal Chroma record count: {final_count}"
    )

    assert final_count == 3

    print("\n")
    print("=" * 60)
    print("PHASE 6 INGESTION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()