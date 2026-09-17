import shutil

from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


def main():
    persist_directory = "data/chroma_modified_test"
    collection_name = "modified_test"

    # Start with a clean database
    shutil.rmtree(
        persist_directory,
        ignore_errors=True,
    )

    # --------------------------------------------------
    # Create original document
    # --------------------------------------------------

    original_chunks = [
        Chunk(
            text="Python is a programming language.",
            chunk_id=0,
            source="document.txt",
        ),
        Chunk(
            text="Python is widely used in AI.",
            chunk_id=1,
            source="document.txt",
        ),
    ]

    original_text = "\n".join(
        chunk.text
        for chunk in original_chunks
    )

    original_hash = calculate_content_hash(
        original_text
    )

    print("Original hash:")
    print(original_hash)

    # --------------------------------------------------
    # Create embeddings
    # --------------------------------------------------

    embedder = Embedder()

    embeddings = embedder.embed_texts(
        [chunk.text for chunk in original_chunks]
    )

    # --------------------------------------------------
    # Create vector store
    # --------------------------------------------------

    store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # --------------------------------------------------
    # Document should not exist initially
    # --------------------------------------------------

    stored_hash = store.get_document_hash(
        "test_document"
    )

    print("\nBefore ingestion:")
    print("Stored hash:", stored_hash)

    assert stored_hash is None

    # --------------------------------------------------
    # Ingest original document
    # --------------------------------------------------

    store.add_chunks(
        chunks=original_chunks,
        embeddings=embeddings,
        document_id="test_document",
        content_hash=original_hash,
    )

    # --------------------------------------------------
    # Retrieve stored hash
    # --------------------------------------------------

    stored_hash = store.get_document_hash(
        "test_document"
    )

    print("\nAfter ingestion:")
    print("Stored hash:", stored_hash)

    assert stored_hash == original_hash

    # --------------------------------------------------
    # Simulate modified document
    # --------------------------------------------------

    modified_text = (
        "Python is a programming language.\n"
        "Python is widely used in machine learning."
    )

    modified_hash = calculate_content_hash(
        modified_text
    )

    print("\nModified hash:")
    print(modified_hash)

    # The modified document must have a different hash
    assert modified_hash != original_hash

    # --------------------------------------------------
    # Compare hashes
    # --------------------------------------------------

    print("\nDocument status:")

    if stored_hash == modified_hash:
        print("UNCHANGED")
    else:
        print("MODIFIED")

    assert stored_hash != modified_hash

    print("\nModified document detection test passed.")


if __name__ == "__main__":
    main()