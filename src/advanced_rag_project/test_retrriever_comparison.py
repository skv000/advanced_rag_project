from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.retrieval.retriever import Retriever
from advanced_rag_project.retrieval.chroma_retriever import ChromaRetriever


def main():
    document_path = "data/documents/rag_test_document.txt"

    text = load_text_file(document_path)

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=50,
        source=document_path,
    )

    query = "What is Advaita Vedanta?"

    # --------------------------------------------------
    # Original in-memory retriever
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("ORIGINAL IN-MEMORY RETRIEVER")
    print("=" * 70)

    original_retriever = Retriever(chunks)

    original_results = original_retriever.search(
        query=query,
        top_k=3,
    )

    for chunk, score in original_results:
        print("\n" + "-" * 70)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Similarity: {score:.4f}")
        print(chunk.text)

    # --------------------------------------------------
    # ChromaDB retriever
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CHROMADB RETRIEVER")
    print("=" * 70)

    chroma_retriever = ChromaRetriever(
        persist_directory="data/chroma_ingestion_test",
        collection_name="documents",
    )

    chroma_results = chroma_retriever.search(
        query=query,
        top_k=3,
    )

    for chunk, score in chroma_results:
        print("\n" + "-" * 70)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Similarity: {score:.4f}")
        print(chunk.text)


if __name__ == "__main__":
    main()