import shutil
from pathlib import Path

import chromadb

from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.ingestion.pipeline import IngestionPipeline


def main():
    persist_directory = "data/chroma_modified_ingestion_test"
    collection_name = "modified_ingestion_test"

    test_file = Path(
        "data/documents/modified_test.txt"
    )

    # --------------------------------------------------
    # Clean previous test database
    # --------------------------------------------------

    shutil.rmtree(
        persist_directory,
        ignore_errors=True,
    )

    # --------------------------------------------------
    # Create initial document
    # --------------------------------------------------

    original_text = """Python is a programming language.

Python is widely used in artificial intelligence.

Python can be used to build RAG systems.
"""

    test_file.write_text(
        original_text,
        encoding="utf-8",
    )

    original_hash = calculate_content_hash(
        original_text
    )

    # --------------------------------------------------
    # Create ingestion pipeline
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print("=" * 60)
    print("FIRST INGESTION")
    print("=" * 60)

    pipeline.ingest_file(
        file_path=str(test_file),
        chunk_size=100,
        chunk_overlap=20,
    )

    # --------------------------------------------------
    # Inspect database
    # --------------------------------------------------

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection = client.get_collection(
        collection_name
    )

    first_count = collection.count()

    print("\nRecords after first ingestion:")
    print(first_count)

    assert first_count > 0

    # Verify original hash
    stored_hash = pipeline.vector_store.get_document_hash(
        "modified_test"
    )

    print("\nStored original hash:")
    print(stored_hash)

    assert stored_hash == original_hash

    # --------------------------------------------------
    # Modify document
    # --------------------------------------------------

    modified_text = """Python is a programming language.

Python is widely used in artificial intelligence.

Python can be used to build advanced RAG systems.

RAG combines retrieval with language generation.
"""

    test_file.write_text(
        modified_text,
        encoding="utf-8",
    )

    modified_hash = calculate_content_hash(
        modified_text
    )

    print("\nOriginal hash:")
    print(original_hash)

    print("\nModified hash:")
    print(modified_hash)

    assert modified_hash != original_hash

    # --------------------------------------------------
    # Second ingestion
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SECOND INGESTION - MODIFIED DOCUMENT")
    print("=" * 60)

    pipeline.ingest_file(
        file_path=str(test_file),
        chunk_size=100,
        chunk_overlap=20,
    )

    # --------------------------------------------------
    # Verify replacement
    # --------------------------------------------------

    final_count = collection.count()

    print("\nRecords after modified ingestion:")
    print(final_count)

    # New version should exist
    final_hash = pipeline.vector_store.get_document_hash(
        "modified_test"
    )

    print("\nFinal stored hash:")
    print(final_hash)

    assert final_hash == modified_hash

    # Make sure old hash is gone
    old_hash_exists = pipeline.vector_store.document_exists(
        original_hash
    )

    print("\nOld content hash exists:")
    print(old_hash_exists)

    assert old_hash_exists is False

    # Make sure new hash exists
    new_hash_exists = pipeline.vector_store.document_exists(
        modified_hash
    )

    print("\nNew content hash exists:")
    print(new_hash_exists)

    assert new_hash_exists is True

    # --------------------------------------------------
    # Inspect final documents
    # --------------------------------------------------

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    print("\nFinal stored documents:")

    for document, metadata in zip(
        result["documents"],
        result["metadatas"],
    ):
        print("\nDocument:")
        print(document)

        print("Metadata:")
        print(metadata)

        assert metadata["content_hash"] == modified_hash

    print("\nModified document replacement test passed.")


if __name__ == "__main__":
    main()