from pathlib import Path
import shutil

from advanced_rag_project.ingestion.pipeline import (
    IngestionPipeline,
)


TEST_DIRECTORY = Path(
    "data/ingestion_stats_test"
)

TEST_CHROMA = (
    "data/chroma_ingestion_stats_test"
)


def print_stats(stats):
    """
    Display ingestion statistics.
    """

    print("\n")
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(
        f"Documents scanned : {stats.scanned}"
    )

    print(
        f"New               : {stats.new}"
    )

    print(
        f"Unchanged         : {stats.unchanged}"
    )

    print(
        f"Modified          : {stats.modified}"
    )

    print(
        f"Deleted           : {stats.deleted}"
    )

    print(
        f"Duplicates        : {stats.duplicates}"
    )

    print(
        f"Failed            : {stats.failed}"
    )

    print(
        f"Total processed   : "
        f"{stats.total_processed}"
    )

    print(
        f"Has errors        : "
        f"{stats.has_errors}"
    )

    print("=" * 60)


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
    # Create documents
    # --------------------------------------------------

    new_file = (
        TEST_DIRECTORY / "new.txt"
    )

    unchanged_file = (
        TEST_DIRECTORY / "unchanged.txt"
    )

    modified_file = (
        TEST_DIRECTORY / "modified.txt"
    )

    duplicate_file = (
        TEST_DIRECTORY / "duplicate.txt"
    )

    deleted_file = (
        TEST_DIRECTORY / "deleted.txt"
    )

    # Unsupported file
    unsupported_file = (
        TEST_DIRECTORY / "unsupported.pdf"
    )

    unchanged_file.write_text(
        "This is an unchanged document.",
        encoding="utf-8",
    )

    modified_file.write_text(
        "This is the original modified document.",
        encoding="utf-8",
    )

    deleted_file.write_text(
        "This document will be deleted.",
        encoding="utf-8",
    )

    # The contents don't need to be a real PDF.
    # It only exists to test extension filtering.
    unsupported_file.write_text(
        "This file should never be ingested.",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Create pipeline
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=TEST_CHROMA,
        collection_name="stats_test",
    )

    # --------------------------------------------------
    # Initial ingestion
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("INITIAL INGESTION")
    print("=" * 60)

    first_stats = pipeline.ingest_directory(
        directory=str(TEST_DIRECTORY)
    )

    print_stats(first_stats)

    # --------------------------------------------------
    # Verify initial state
    # --------------------------------------------------

    assert first_stats.scanned == 3

    assert first_stats.new == 3

    assert first_stats.unchanged == 0

    assert first_stats.modified == 0

    assert first_stats.deleted == 0

    assert first_stats.duplicates == 0

    assert first_stats.failed == 0

    assert first_stats.total_processed == 3

    assert first_stats.has_errors is False

    # --------------------------------------------------
    # Verify unsupported file was ignored
    # --------------------------------------------------

    assert unsupported_file.exists()

    # --------------------------------------------------
    # Add a new file
    # --------------------------------------------------

    new_file.write_text(
        "This is a brand new document.",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Create duplicate content
    # --------------------------------------------------

    duplicate_file.write_text(
        "This is a brand new document.",
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Modify existing document
    # --------------------------------------------------

    modified_file.write_text(
        """
This is the modified version of the document.

The content has changed.
""".strip(),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Delete existing document
    # --------------------------------------------------

    deleted_file.unlink()

    # --------------------------------------------------
    # Second ingestion
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SECOND INGESTION")
    print("=" * 60)

    second_stats = pipeline.ingest_directory(
        directory=str(TEST_DIRECTORY)
    )

    print_stats(second_stats)

    # --------------------------------------------------
    # Verify statistics
    # --------------------------------------------------

    assert second_stats.scanned == 4

    assert second_stats.new == 1

    assert second_stats.unchanged == 1

    assert second_stats.modified == 1

    assert second_stats.deleted == 1

    assert second_stats.duplicates == 1

    assert second_stats.failed == 0

    assert second_stats.total_processed == 4

    assert second_stats.has_errors is False

    # --------------------------------------------------
    # Verify dictionary conversion
    # --------------------------------------------------

    stats_dict = second_stats.to_dict()

    assert stats_dict["scanned"] == 4

    assert stats_dict["new"] == 1

    assert stats_dict["unchanged"] == 1

    assert stats_dict["modified"] == 1

    assert stats_dict["deleted"] == 1

    assert stats_dict["duplicates"] == 1

    assert stats_dict["failed"] == 0

    # --------------------------------------------------
    # Cleanup filesystem test data
    # --------------------------------------------------

    shutil.rmtree(
        TEST_DIRECTORY
    )

    print("\n")
    print("=" * 60)
    print("INGESTION STATISTICS TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()