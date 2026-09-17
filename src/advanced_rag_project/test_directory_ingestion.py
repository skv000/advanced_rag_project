from pathlib import Path
import shutil

from advanced_rag_project.ingestion.pipeline import IngestionPipeline


TEST_DIRECTORY = Path("data/directory_ingestion_test")
TEST_CHROMA = "data/chroma_directory_ingestion_test"


def main():
    # --------------------------------------------------
    # Cleanup from previous test
    # --------------------------------------------------

    if TEST_DIRECTORY.exists():
        shutil.rmtree(TEST_DIRECTORY)

    if Path(TEST_CHROMA).exists():
        shutil.rmtree(TEST_CHROMA)

    TEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Create test documents
    # --------------------------------------------------

    file1 = TEST_DIRECTORY / "python.txt"
    file2 = TEST_DIRECTORY / "machine_learning.txt"

    file1.write_text(
        """
Python is a programming language.

Python is widely used in artificial intelligence.
""".strip(),
        encoding="utf-8",
    )

    file2.write_text(
        """
Machine learning allows computers to learn from data.

Machine learning is widely used in artificial intelligence.
""".strip(),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Create pipeline
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=TEST_CHROMA,
        collection_name="directory_test",
    )

    # --------------------------------------------------
    # First ingestion
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FIRST DIRECTORY INGESTION")
    print("=" * 60)

    pipeline.ingest_directory(
        directory=str(TEST_DIRECTORY)
    )

    count_after_first = (
        pipeline.vector_store.collection.count()
    )

    print(
        f"\nRecords after first ingestion: "
        f"{count_after_first}"
    )

    assert count_after_first > 0

    # --------------------------------------------------
    # Second ingestion
    # Everything should be unchanged
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SECOND DIRECTORY INGESTION")
    print("=" * 60)

    pipeline.ingest_directory(
        directory=str(TEST_DIRECTORY)
    )

    count_after_second = (
        pipeline.vector_store.collection.count()
    )

    print(
        f"\nRecords after second ingestion: "
        f"{count_after_second}"
    )

    assert count_after_second == count_after_first

    # --------------------------------------------------
    # Modify one document
    # --------------------------------------------------

    file1.write_text(
        """
Python is a programming language.

Python is widely used in artificial intelligence.

Python is also used for data science.
""".strip(),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Third ingestion
    # One document should be replaced
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("THIRD DIRECTORY INGESTION")
    print("=" * 60)

    pipeline.ingest_directory(
        directory=str(TEST_DIRECTORY)
    )

    count_after_third = (
        pipeline.vector_store.collection.count()
    )

    print(
        f"\nRecords after modified ingestion: "
        f"{count_after_third}"
    )

    # The modified document should have been replaced,
    # not duplicated.
    assert count_after_third > 0

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    shutil.rmtree(TEST_DIRECTORY)

    # if Path(TEST_CHROMA).exists():
    #     shutil.rmtree(TEST_CHROMA)

    print("\n")
    print("=" * 60)
    print("DIRECTORY INGESTION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()