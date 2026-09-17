from pathlib import Path
import shutil

from advanced_rag_project.documents.identifiers import (
    create_document_id,
)
from advanced_rag_project.ingestion.pipeline import (
    IngestionPipeline,
)


TEST_DIRECTORY = Path(
    "data/deleted_ingestion_test"
)

TEST_CHROMA = (
    "data/chroma_deleted_ingestion_test"
)


def main():

    # --------------------------------------------------
    # Cleanup from previous test
    # --------------------------------------------------

    if TEST_DIRECTORY.exists():

        shutil.rmtree(
            TEST_DIRECTORY
        )

    if Path(TEST_CHROMA).exists():

        shutil.rmtree(
            TEST_CHROMA
        )

    # --------------------------------------------------
    # Create test directory
    # --------------------------------------------------

    TEST_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Create test documents
    # --------------------------------------------------

    file1 = (
        TEST_DIRECTORY / "python.txt"
    )

    file2 = (
        TEST_DIRECTORY / "machine_learning.txt"
    )

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
    # Create ingestion pipeline
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=TEST_CHROMA,
        collection_name="deleted_test",
    )

    # --------------------------------------------------
    # FIRST INGESTION
    # --------------------------------------------------

    print("\n")

    print(
        "=" * 60
    )

    print(
        "FIRST INGESTION"
    )

    print(
        "=" * 60
    )

    pipeline.ingest_directory(
        directory=str(
            TEST_DIRECTORY
        )
    )

    initial_count = (
        pipeline.vector_store.collection.count()
    )

    print(
        f"\nInitial Chroma record count: "
        f"{initial_count}"
    )

    assert initial_count == 2

    # --------------------------------------------------
    # Verify both document IDs exist
    # --------------------------------------------------

    stored_ids = (
        pipeline.vector_store.get_document_ids()
    )

    python_id = create_document_id(
        str(file1)
    )

    machine_learning_id = (
        create_document_id(
            str(file2)
        )
    )

    assert python_id in stored_ids

    assert (
        machine_learning_id
        in stored_ids
    )

    print(
        "\nBoth documents exist in ChromaDB."
    )

    # --------------------------------------------------
    # DELETE python.txt from filesystem
    # --------------------------------------------------

    print("\n")

    print(
        "=" * 60
    )

    print(
        "DELETE PYTHON DOCUMENT FROM FILESYSTEM"
    )

    print(
        "=" * 60
    )

    file1.unlink()

    print(
        f"Deleted: {file1}"
    )

    # --------------------------------------------------
    # SECOND INGESTION
    # --------------------------------------------------

    print("\n")

    print(
        "=" * 60
    )

    print(
        "SECOND INGESTION"
    )

    print(
        "=" * 60
    )

    pipeline.ingest_directory(
        directory=str(
            TEST_DIRECTORY
        )
    )

    # --------------------------------------------------
    # Verify deleted document was removed
    # --------------------------------------------------

    final_ids = (
        pipeline.vector_store.get_document_ids()
    )

    final_count = (
        pipeline.vector_store.collection.count()
    )

    print(
        f"\nFinal Chroma record count: "
        f"{final_count}"
    )

    print(
        f"Final document IDs: "
        f"{final_ids}"
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert python_id not in final_ids

    assert (
        machine_learning_id
        in final_ids
    )

    assert final_count == 1

    print(
        "\nDeleted document successfully removed."
    )

    # --------------------------------------------------
    # Cleanup filesystem test data
    # --------------------------------------------------

    shutil.rmtree(
        TEST_DIRECTORY
    )

    print("\n")

    print(
        "=" * 60
    )

    print(
        "DELETED DOCUMENT TEST PASSED"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()