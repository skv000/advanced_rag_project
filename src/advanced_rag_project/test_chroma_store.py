from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


def main():
    chunks = [
        Chunk(
            text="Python is a programming language.",
            chunk_id=0,
            source="test.txt",
        ),
        Chunk(
            text="RAG combines retrieval with language generation.",
            chunk_id=1,
            source="test.txt",
        ),
        Chunk(
            text="Vector embeddings represent semantic meaning.",
            chunk_id=2,
            source="test.txt",
        ),
    ]

    # Create embeddings
    embedder = Embedder()

    embeddings = embedder.embed_texts(
        [chunk.text for chunk in chunks]
    )

    # Create content hash
    document_text = "\n".join(
        chunk.text
        for chunk in chunks
    )

    content_hash = calculate_content_hash(document_text)

    print("\nContent Hash:")
    print(content_hash)

    # Create vector store
    store = ChromaVectorStore(
        persist_directory="data/chroma_test",
        collection_name="test_collection",
    )

    # Add chunks
    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id="test_document",
        content_hash=content_hash,
    )

    # Create query embedding
    query = "How does retrieval augmented generation work?"

    query_embedding = embedder.embed_text(query)

    # Search
    results = store.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    print("\nSearch Results:")
    print(results)


if __name__ == "__main__":
    main()