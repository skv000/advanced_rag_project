import chromadb

from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


def main():
    chunks = [
        Chunk(
            text="Python is useful for AI development.",
            chunk_id=0,
            source="metadata_test.txt",
        ),
        Chunk(
            text="RAG systems retrieve relevant information.",
            chunk_id=1,
            source="metadata_test.txt",
        ),
    ]

    embedder = Embedder()

    embeddings = embedder.embed_texts(
        [chunk.text for chunk in chunks]
    )

    document_text = "\n".join(
        chunk.text
        for chunk in chunks
    )

    content_hash = calculate_content_hash(document_text)

    store = ChromaVectorStore(
        persist_directory="data/chroma_metadata_test",
        collection_name="metadata_test",
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="metadata_document",
        content_hash=content_hash,
    )

    # Open the same persistent ChromaDB
    client = chromadb.PersistentClient(
        path="data/chroma_metadata_test"
    )

    collection = client.get_collection(
        "metadata_test"
    )

    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    print("\nStored IDs:")
    print(result["ids"])

    print("\nStored Metadata:")

    for metadata in result["metadatas"]:
        print(metadata)

        assert metadata["document_id"] == "metadata_document"
        assert metadata["content_hash"] == content_hash
        assert "chunk_id" in metadata
        assert "source" in metadata

    print("\nContent hash metadata test passed.")


if __name__ == "__main__":
    main()