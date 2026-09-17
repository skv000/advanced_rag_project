import shutil

import chromadb

from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.ingestion.pipeline import IngestionPipeline


def main():
    persist_directory = "data/chroma_ingestion_duplicate_test"
    collection_name = "ingestion_duplicate_test"

    # Start with a completely clean database
    shutil.rmtree(
        persist_directory,
        ignore_errors=True,
    )

    file_path = "data/documents/hindu_philosophy.txt"

    # Calculate the expected hash independently
    text = load_text_file(file_path)
    expected_hash = calculate_content_hash(text)

    print("=" * 60)
    print("EXPECTED DOCUMENT HASH")
    print("=" * 60)
    print(expected_hash)

    # Create pipeline
    pipeline = IngestionPipeline(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print("\n" + "=" * 60)
    print("BEFORE FIRST INGESTION")
    print("=" * 60)

    exists_before = pipeline.vector_store.document_exists(
        expected_hash
    )

    print("Document exists:", exists_before)

    assert exists_before is False

    print("\n" + "=" * 60)
    print("FIRST INGESTION")
    print("=" * 60)

    pipeline.ingest_file(
        file_path=file_path,
        chunk_size=200,
        chunk_overlap=30,
    )

    # Check using the SAME vector store instance
    exists_after_first = pipeline.vector_store.document_exists(
        expected_hash
    )

    print("\nAfter first ingestion:")
    print("Document exists:", exists_after_first)

    # Inspect Chroma directly
    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection = client.get_collection(
        collection_name
    )

    print("\nChromaDB record count:")
    print(collection.count())

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    print("\nStored metadata:")

    for metadata in result["metadatas"]:
        print(metadata)

    assert exists_after_first is True

    print("\n" + "=" * 60)
    print("SECOND INGESTION")
    print("=" * 60)

    # Check BEFORE calling ingest_file
    exists_before_second = pipeline.vector_store.document_exists(
        expected_hash
    )

    print("Document exists before second ingestion:")
    print(exists_before_second)

    assert exists_before_second is True

    pipeline.ingest_file(
        file_path=file_path,
        chunk_size=200,
        chunk_overlap=30,
    )

    final_count = collection.count()

    print("\nFinal ChromaDB record count:")
    print(final_count)

    print("\nDuplicate ingestion test passed.")


if __name__ == "__main__":
    main()