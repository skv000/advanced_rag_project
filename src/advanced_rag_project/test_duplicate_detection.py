import chromadb

from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


def main():
    persist_directory = "data/chroma_duplicate_test"
    collection_name = "duplicate_test"

    # Clean persistent database for a fresh test
    import shutil

    shutil.rmtree(
        persist_directory,
        ignore_errors=True,
    )

    # Test document
    chunks = [
        Chunk(
            text="Python is a programming language.",
            chunk_id=0,
            source="test_document.txt",
        ),
        Chunk(
            text="Python is commonly used for AI.",
            chunk_id=1,
            source="test_document.txt",
        ),
    ]

    # Create document content
    document_text = "\n".join(
        chunk.text
        for chunk in chunks
    )

    # Calculate content hash
    content_hash = calculate_content_hash(
        document_text
    )

    print("Content Hash:")
    print(content_hash)

    # Create embeddings
    embedder = Embedder()

    embeddings = embedder.embed_texts(
        [chunk.text for chunk in chunks]
    )

    # Create vector store
    store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # Before ingestion, document should not exist
    exists_before = store.document_exists(
        content_hash
    )

    print("\nBefore ingestion:")
    print("Document exists:", exists_before)

    assert exists_before is False

    # Add document
    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="test_document",
        content_hash=content_hash,
    )

    # After ingestion, document should exist
    exists_after = store.document_exists(
        content_hash
    )

    print("\nAfter ingestion:")
    print("Document exists:", exists_after)

    assert exists_after is True

    # Verify a completely different hash is not found
    different_hash = calculate_content_hash(
        "This is a completely different document."
    )

    different_exists = store.document_exists(
        different_hash
    )

    print("\nDifferent document:")
    print("Document exists:", different_exists)

    assert different_exists is False

    print("\nDuplicate detection test passed.")


if __name__ == "__main__":
    main()