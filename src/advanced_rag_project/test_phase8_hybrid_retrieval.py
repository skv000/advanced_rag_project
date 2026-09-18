from pathlib import Path
import shutil

from advanced_rag_project.documents.chunker import (
    chunk_text,
)
from advanced_rag_project.documents.loader import (
    load_document,
)
from advanced_rag_project.ingestion.pipeline import (
    IngestionPipeline,
)
from advanced_rag_project.retrieval.chroma_retriever import (
    ChromaRetriever,
)
from advanced_rag_project.retrieval.hybrid_retriever import (
    HybridRetriever,
)


TEST_DIR = Path(
    "data/phase8_hybrid_test"
)

CHROMA_DIR = (
    "data/chroma_phase8_test"
)


def create_documents():

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (TEST_DIR / "rag.md").write_text(
        """
        # Retrieval Augmented Generation

        Retrieval Augmented Generation, also called RAG,
        combines document retrieval with language model
        generation.

        A retrieval system finds relevant information
        before the language model generates an answer.
        """.strip(),
        encoding="utf-8",
    )

    (TEST_DIR / "python.txt").write_text(
        """
        Python is a programming language used for
        artificial intelligence, machine learning,
        automation, and web development.
        """.strip(),
        encoding="utf-8",
    )

    (TEST_DIR / "philosophy.txt").write_text(
        """
        Advaita Vedanta is a non-dual philosophical
        tradition concerned with Brahman, Atman,
        and ultimate reality.
        """.strip(),
        encoding="utf-8",
    )


def load_chunks():

    chunks = []

    for path in TEST_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".txt",
            ".md",
        }:
            continue

        document = load_document(path)

        document_chunks = chunk_text(
            document.text,
            chunk_size=500,
            chunk_overlap=50,
            source=document.source,
        )

        chunks.extend(
            document_chunks
        )

    return chunks


def print_results(
    title,
    results,
):

    print("\n")
    print("=" * 60)
    print(title)
    print("=" * 60)

    for index, result in enumerate(
        results,
        start=1,
    ):

        if hasattr(
            result,
            "hybrid_score",
        ):

            print(
                f"{index}. "
                f"hybrid={result.hybrid_score:.4f} "
                f"dense={result.dense_score:.4f} "
                f"keyword={result.keyword_score:.4f}"
            )

        else:

            print(
                f"{index}. "
                f"score={result.similarity:.4f}"
            )

        print(
            f"   source={result.source}"
        )


def main():

    create_documents()

    # --------------------------------------------------
    # Ingestion
    # --------------------------------------------------

    pipeline = IngestionPipeline(
        persist_directory=CHROMA_DIR,
        collection_name="phase8_test",
    )

    print("\n")
    print("=" * 60)
    print("PHASE 8 - INGESTION")
    print("=" * 60)

    stats = pipeline.ingest_directory(
        str(TEST_DIR)
    )

    assert stats.new == 3
    assert stats.failed == 0

    # --------------------------------------------------
    # Load chunks for keyword index
    # --------------------------------------------------

    chunks = load_chunks()

    assert len(chunks) == 3

    # --------------------------------------------------
    # Dense retriever
    # --------------------------------------------------

    dense_retriever = ChromaRetriever(
        persist_directory=CHROMA_DIR,
        collection_name="phase8_test",
    )

    dense_results = dense_retriever.search(
        query="retrieval augmented generation",
        top_k=3,
    )

    assert len(dense_results) > 0

    print_results(
        "DENSE RETRIEVAL",
        dense_results,
    )

    # --------------------------------------------------
    # Keyword retriever
    # --------------------------------------------------

    from advanced_rag_project.retrieval.keyword_retriever import (
        KeywordRetriever,
    )

    keyword_retriever = KeywordRetriever(
        chunks
    )

    keyword_results = keyword_retriever.search(
        query="retrieval augmented generation",
        top_k=3,
    )

    assert len(keyword_results) > 0

    print_results(
        "KEYWORD RETRIEVAL",
        keyword_results,
    )

    # --------------------------------------------------
    # Hybrid
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        chunks=chunks,
        persist_directory=CHROMA_DIR,
        collection_name="phase8_test",
    )

    hybrid_results = hybrid_retriever.search(
        query="retrieval augmented generation",
        top_k=3,
    )

    assert len(hybrid_results) > 0

    print_results(
        "HYBRID RETRIEVAL",
        hybrid_results,
    )

    # --------------------------------------------------
    # Ranking
    # --------------------------------------------------

    scores = [
        result.hybrid_score
        for result in hybrid_results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # --------------------------------------------------
    # Weight validation
    # --------------------------------------------------

    try:

        hybrid_retriever.search(
            query="test",
            top_k=3,
            dense_weight=0,
            keyword_weight=0,
        )

        raise AssertionError(
            "Zero total weight should fail."
        )

    except ValueError:

        print(
            "\nWeight validation passed."
        )

    # --------------------------------------------------
    # Exact keyword query
    # --------------------------------------------------

    exact_results = (
        hybrid_retriever.search(
            query="Brahman Atman",
            top_k=3,
            dense_weight=0.4,
            keyword_weight=0.6,
        )
    )

    assert len(exact_results) > 0

    print_results(
        "KEYWORD-HEAVY HYBRID SEARCH",
        exact_results,
    )

    # --------------------------------------------------
    # Complete
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PHASE 8 HYBRID RETRIEVAL TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()