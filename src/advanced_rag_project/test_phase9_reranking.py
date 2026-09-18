from pathlib import Path
import shutil

from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.documents.loader import load_document
from advanced_rag_project.ingestion.pipeline import IngestionPipeline
from advanced_rag_project.retrieval.reranking_retriever import (
    RerankingRetriever,
)
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


BASE_DIR = Path("data/test_phase9")
DOCUMENT_DIR = BASE_DIR / "documents"

CHROMA_DIR = "data/chroma_phase9"
COLLECTION_NAME = "phase9_test"


def create_test_documents():
    """
    Create controlled documents for the Phase 9 integration test.
    """

    if DOCUMENT_DIR.exists():
        shutil.rmtree(DOCUMENT_DIR)

    DOCUMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (DOCUMENT_DIR / "ai.txt").write_text(
        """
Artificial intelligence is the field of building systems
that can perform tasks requiring human-like reasoning.

Machine learning allows systems to learn patterns from data.
Deep learning uses neural networks to model complex patterns.

Retrieval augmented generation combines information retrieval
with language models to improve factual responses.
""".strip(),
        encoding="utf-8",
    )

    (DOCUMENT_DIR / "rag.txt").write_text(
        """
Retrieval augmented generation, commonly called RAG,
combines a retrieval system with a generative language model.

A RAG system first retrieves relevant documents or chunks.
The retrieved context is then supplied to the language model.

Dense vector retrieval uses embeddings to find semantically
similar text.

Keyword retrieval can identify exact terms in documents.
Hybrid retrieval combines semantic and lexical signals.

A reranker can then examine the query and retrieved passages
together and produce a more accurate relevance ordering.
""".strip(),
        encoding="utf-8",
    )

    (DOCUMENT_DIR / "philosophy.txt").write_text(
        """
Advaita Vedanta is a school of Indian philosophy.

Advaita Vedanta teaches that ultimate reality is non-dual.
Brahman is described as the fundamental reality.

The relationship between Atman and Brahman is an important
concept in Advaita Vedanta.

The Bhagavad Gita discusses knowledge, action, and devotion.
""".strip(),
        encoding="utf-8",
    )


def build_chunks():
    """
    Load the same documents and create chunks using the
    same chunking configuration used during ingestion.
    """

    chunks = []

    for path in DOCUMENT_DIR.glob("*.txt"):
        document = load_document(path)

        document_chunks = chunk_text(
            document.text,
            chunk_size=180,
            chunk_overlap=40,
            source=document.source,
        )

        chunks.extend(document_chunks)

    return chunks


def main():
    print("=" * 60)
    print("PHASE 9 — CROSS-ENCODER RERANKING TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Create test documents
    # ---------------------------------------------------------

    create_test_documents()

    # ---------------------------------------------------------
    # 2. Ingest documents using the EXISTING pipeline API
    # ---------------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    stats = pipeline.ingest_directory(
        str(DOCUMENT_DIR),
        chunk_size=180,
        chunk_overlap=40,
    )

    print("\nIngestion statistics:")
    print(stats.to_dict())

    assert stats.failed == 0
    assert stats.new + stats.unchanged + stats.modified == 3

    # ---------------------------------------------------------
    # 3. Verify Chroma contains multiple chunks
    # ---------------------------------------------------------

    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    vector_count = vector_store.collection.count()

    print(f"\nChroma vector count: {vector_count}")

    assert vector_count > 3

    # ---------------------------------------------------------
    # 4. Build chunks for keyword retrieval
    # ---------------------------------------------------------

    chunks = build_chunks()

    print(f"Total chunks available: {len(chunks)}")

    assert len(chunks) > 3

    # ---------------------------------------------------------
    # 5. Create reranking retriever
    # ---------------------------------------------------------

    retriever = RerankingRetriever(
        chunks=chunks,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        batch_size=8,
    )

    # ---------------------------------------------------------
    # 6. Run candidate retrieval + reranking
    # ---------------------------------------------------------

    query = (
        "How does RAG retrieve information before "
        "generating an answer?"
    )

    print(f"\nQuery: {query}")

    candidate_k = 6
    final_top_k = 3

    print(
        f"\nCandidate K : {candidate_k}"
    )

    print(
        f"Final Top K : {final_top_k}"
    )

    print(
        "\nRunning hybrid retrieval + cross-encoder reranking..."
    )

    results = retriever.search(
        query=query,
        candidate_k=candidate_k,
        top_k=final_top_k,
        dense_weight=0.7,
        keyword_weight=0.3,
    )

    # ---------------------------------------------------------
    # 7. Display results
    # ---------------------------------------------------------

    print("\nFinal reranked results:")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 60)

        print(f"Rank          : {index}")
        print(
            f"Rerank score  : "
            f"{result.rerank_score:.4f}"
        )
        print(
            f"Hybrid score  : "
            f"{result.hybrid_score:.4f}"
        )
        print(
            f"Dense score   : "
            f"{result.dense_score:.4f}"
        )
        print(
            f"Keyword score : "
            f"{result.keyword_score:.4f}"
        )
        print(
            f"Source        : "
            f"{result.source}"
        )
        print(
            f"Chunk ID      : "
            f"{result.chunk_id}"
        )

        print(
            f"Text          : "
            f"{result.chunk.text[:300]}"
        )

    # ---------------------------------------------------------
    # 8. Validate result count
    # ---------------------------------------------------------

    assert len(results) <= final_top_k

    # ---------------------------------------------------------
    # 9. Validate reranker scores
    # ---------------------------------------------------------

    scores = [
        result.rerank_score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # ---------------------------------------------------------
    # 10. Validate result structure
    # ---------------------------------------------------------

    for result in results:

        assert isinstance(
            result.rerank_score,
            float,
        )

        assert isinstance(
            result.hybrid_score,
            float,
        )

        assert isinstance(
            result.dense_score,
            float,
        )

        assert isinstance(
            result.keyword_score,
            float,
        )

        assert result.source

        assert result.chunk.text

    # ---------------------------------------------------------
    # 11. Validate query
    # ---------------------------------------------------------

    try:
        retriever.search("")

        raise AssertionError(
            "Empty query should raise ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------------------
    # 12. Validate candidate_k
    # ---------------------------------------------------------

    try:
        retriever.search(
            query=query,
            candidate_k=0,
            top_k=3,
        )

        raise AssertionError(
            "candidate_k=0 should raise ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------------------
    # 13. Validate top_k <= candidate_k
    # ---------------------------------------------------------

    try:
        retriever.search(
            query=query,
            candidate_k=2,
            top_k=3,
        )

        raise AssertionError(
            "top_k > candidate_k should raise ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------------------
    # 14. Success
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 9 RERANKING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()