from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.retrieval.retriever import Retriever
from advanced_rag_project.rag.context_builder import build_context


def main():
    print("=" * 50)
    print("Context Builder Test")
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

    query = "what does advaita vedanta teach?"

    print("\n" + "=" * 50)
    print(f"Query: {query}")
    print("=" * 50)

    results = retriever.search(
        query=query,
        top_k=2,
    )

    context = build_context(
        results=results
    )

    print("\n" + "=" * 50)
    print("RAG CONTEXT")
    print("=" * 50)
    print(context)


if __name__ == "__main__":
    main()
