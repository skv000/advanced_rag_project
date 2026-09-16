from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.retrieval.retriever import Retriever


def main():
    print("=" * 50)
    print("Retriever Test")
    print("=" * 50)

    document = load_text_file(
        "data/documents/hindu_philosophy.txt"
    )

    chunks = chunk_text(
        text=document,
        chunk_size=200,
        chunk_overlap=30,
        source="hindu_philosophy.txt",
    )

    print(f"Number of chunks: {len(chunks)}")

    retriever = Retriever(
        chunks=chunks
    )

    query = "what advaita vedanta teach?"

    print("\n" + "=" * 50)
    print(f"Query: {query}")
    print("=" * 50)

    results = retriever.search(
        query=query,
        top_k=2,
    )

    for index, (chunk, score) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 50)
        print(f"Result: {index}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Similarity: {score:.5f}")
        print("-" * 50)
        print(chunk.text)


if __name__ == "__main__":
    main()
