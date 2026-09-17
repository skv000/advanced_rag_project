import shutil

import chromadb

from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


def main():
    persist_directory = "data/chroma_deletion_test"
    collection_name = "deletion_test"

    # Start with a clean database
    shutil.rmtree(
        persist_directory,
        ignore_errors=True,
    )

    # --------------------------------------------------
    # Create document A
    # --------------------------------------------------

    chunks_a = [
        Chunk(
            text="Document A first chunk.",
            chunk_id=0,
            source="document_a.txt",
        ),
        Chunk(
            text="Document A second chunk.",
            chunk_id=1,
            source="document_a.txt",
        ),
        Chunk(
            text="Document A third chunk.",
            chunk_id=2,
            source="document_a.txt",
        ),
    ]

    document_a_text = "\n".join(
        chunk.text
        for chunk in chunks_a
    )

    document_a_hash = calculate_content_hash(
        document_a_text
    )

    # --------------------------------------------------
    # Create document B
    # --------------------------------------------------

    chunks_b = [
        Chunk(
            text="Document B first chunk.",
            chunk_id=0,
            source="document_b.txt",
        ),
        Chunk(
            text="Document B second chunk.",
            chunk_id=1,
            source="document_b.txt",
        ),
    ]

    document_b_text = "\n".join(
        chunk.text
        for chunk in chunks_b
    )

    document_b_hash = calculate_content_hash(
        document_b_text
    )

    # --------------------------------------------------
    # Create embeddings
    # --------------------------------------------------

    embedder = Embedder()

    embeddings_a = embedder.embed_texts(
        [chunk.text for chunk in chunks_a]
    )

    embeddings_b = embedder.embed_texts(
        [chunk.text for chunk in chunks_b]
    )

    # --------------------------------------------------
    # Create vector store
    # --------------------------------------------------

    store = ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # --------------------------------------------------
    # Add both documents
    # --------------------------------------------------

    store.add_chunks(
        chunks=chunks_a,
        embeddings=embeddings_a,
        document_id="document_a",
        content_hash=document_a_hash,
    )

    store.add_chunks(
        chunks=chunks_b,
        embeddings=embeddings_b,
        document_id="document_b",
        content_hash=document_b_hash,
    )

    # --------------------------------------------------
    # Verify initial state
    # --------------------------------------------------

    client = chromadb.PersistentClient(
        path=persist_directory
    )

    collection = client.get_collection(
        collection_name
    )

    initial_count = collection.count()

    print("Initial record count:")
    print(initial_count)

    assert initial_count == 5

    print("\nDocument A exists:")
    print(store.document_exists(document_a_hash))

    print("\nDocument B exists:")
    print(store.document_exists(document_b_hash))

    assert store.document_exists(document_a_hash) is True
    assert store.document_exists(document_b_hash) is True

    # --------------------------------------------------
    # Delete document A
    # --------------------------------------------------

    print("\nDeleting document A...")

    store.delete_document(
        document_id="document_a"
    )

    # --------------------------------------------------
    # Verify document A was deleted
    # --------------------------------------------------

    final_count = collection.count()

    print("\nFinal record count:")
    print(final_count)

    assert final_count == 2

    # Document A should no longer exist
    document_a_exists = store.document_exists(
        document_a_hash
    )

    print("\nDocument A exists after deletion:")
    print(document_a_exists)

    assert document_a_exists is False

    # Document B must still exist
    document_b_exists = store.document_exists(
        document_b_hash
    )

    print("\nDocument B exists after deletion:")
    print(document_b_exists)

    assert document_b_exists is True

    print("\nDocument deletion test passed.")


if __name__ == "__main__":
    main()