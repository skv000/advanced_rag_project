from advanced_rag_project.retrieval.chroma_retriever import ChromaRetriever


def main():
    retriever = ChromaRetriever(
        persist_directory="data/chroma_ingestion_test",
        collection_name="documents",
    )

    query = "What is Advaita Vedanta?"

    results = retriever.search(
        query=query,
        top_k=3,
    )

    print("\nChroma Retriever Results:")

    for chunk, score in results:
        print("\n" + "=" * 60)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Similarity: {score:.4f}")
        print(f"Text: {chunk.text}")


if __name__ == "__main__":
    main()