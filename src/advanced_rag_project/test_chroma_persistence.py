from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


def main():
    embedder = Embedder()

    store = ChromaVectorStore(
        persist_directory="data/chroma_test",
        collection_name="test_collection",
    )

    query = "What is retrieval augmented generation?"

    query_embedding = embedder.embed_text(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    print("\nResults from persisted ChromaDB:")
    print(results)


if __name__ == "__main__":
    main()