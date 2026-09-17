from pathlib import Path
import shutil

from advanced_rag_project.documents.identifiers import (
    create_document_id,
)
from advanced_rag_project.ingestion.pipeline import (
    IngestionPipeline,
)


TEST_DIRECTORY = Path(
    "data/recursive_ingestion_test"
)

TEST_CHROMA = (
    "data/chroma_recursive_ingestion_test"
)


def main():

    # --------------------------------------------------
    # Cleanup previous test
    # --------------------------------------------------

    if TEST_DIRECTORY.exists():
        shutil.rmtree(TEST_DIRECTORY)

    if Path(TEST_CHROMA).exists():
        shutil.rmtree(TEST_CHROMA)

    # --------------------------------------------------
    # Create nested directories
    # --------------------------------------------------

    root_directory = TEST_DIRECTORY

    philosophy_directory = (
        root_directory / "philosophy"
    )

    research_directory = (
        root_directory / "research"
    )

    philosophy_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    research_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Create supported TXT documents
    # --------------------------------------------------

    root_file = (
        root_directory / "python.txt"
    )

    philosophy_file = (
        philosophy_directory / "vedanta.txt"
    )

    research_file = (
        research_directory / "rag.txt"
    )

    # --------------------------------------------------
    # Create unsupported files
    # --------------------------------------------------

    markdown_file = (
        research_directory / "notes.md"
    )

    pdf_file = (
        research_directory / "paper.pdf"
    )

    # --------------------------------------------------
    # Write test documents
    # --------------------------------------------------

    root_file.write_text(
        "Python is a programming language.",
        encoding="utf-8",
    )

    philosophy_file.write_text(
        "Vedanta is a school of Indian philosophy.",
        encoding="utf-8",
    )

    research_file.write_text(
        "Retrieval augmented generation combines "
        "retrieval with generation.",
        encoding="utf-8",
    )

    markdown_file.write_text(
        "# Notes\n\nThis Markdown file should "
        "not be ingested yet.",
        encoding="utf-8",
    )

    pdf_file.write_text(
        "This PDF should be ignored.",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Create pipeline
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=TEST_CHROMA,
        collection_name="recursive_test",
    )

    # --------------------------------------------------
    # First ingestion
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FIRST RECURSIVE INGESTION")
    print("=" * 60)

    stats = pipeline.ingest_directory(
        directory=str(TEST_DIRECTORY)
    )

    # --------------------------------------------------
    # Verify statistics
    # --------------------------------------------------

    assert stats.scanned == 3

    assert stats.new == 3

    assert stats.unchanged == 0

    assert stats.modified == 0

    assert stats.deleted == 0

    assert stats.duplicates == 0

    assert stats.failed == 0

    # --------------------------------------------------
    # Verify document IDs
    # --------------------------------------------------

    stored_ids = (
        pipeline.vector_store.get_document_ids()
    )

    root_id = create_document_id(
        str(root_file)
    )

    philosophy_id = create_document_id(
        str(philosophy_file)
    )

    research_id = create_document_id(
        str(research_file)
    )

    markdown_id = create_document_id(
        str(markdown_file)
    )

    pdf_id = create_document_id(
        str(pdf_file)
    )

    # --------------------------------------------------
    # Supported nested documents exist
    # --------------------------------------------------

    assert root_id in stored_ids

    assert philosophy_id in stored_ids

    assert research_id in stored_ids

    # --------------------------------------------------
    # Unsupported files do not exist
    # --------------------------------------------------

    assert markdown_id not in stored_ids

    assert pdf_id not in stored_ids

    # --------------------------------------------------
    # Verify Chroma count
    # --------------------------------------------------

    count = (
        pipeline.vector_store.collection.count()
    )

    assert count == 3

    print(
        f"\nChroma record count: {count}"
    )

    print(
        "Root document successfully ingested."
    )

    print(
        "Nested documents successfully ingested."
    )

    print(
        "Unsupported files successfully ignored."
    )

    # --------------------------------------------------
    # Second ingestion
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SECOND RECURSIVE INGESTION")
    print("=" * 60)

    second_stats = pipeline.ingest_directory(
        directory=str(TEST_DIRECTORY)
    )

    # --------------------------------------------------
    # Verify second ingestion
    # --------------------------------------------------

    assert second_stats.scanned == 3

    assert second_stats.new == 0

    assert second_stats.unchanged == 3

    assert second_stats.modified == 0

    assert second_stats.deleted == 0

    assert second_stats.duplicates == 0

    assert second_stats.failed == 0

    final_count = (
        pipeline.vector_store.collection.count()
    )

    assert final_count == 3

    print(
        f"\nFinal Chroma record count: "
        f"{final_count}"
    )

    # --------------------------------------------------
    # Cleanup filesystem test data
    # --------------------------------------------------

    shutil.rmtree(
        TEST_DIRECTORY
    )

    print("\n")
    print("=" * 60)
    print("RECURSIVE INGESTION TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()