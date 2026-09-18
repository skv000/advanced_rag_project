import math
import re
from collections import Counter

from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.retrieval.models import RetrievalResult


class KeywordRetriever:
    """
    Lightweight TF-IDF based keyword retriever.

    This implementation is intentionally simple so that
    the retrieval mechanics remain understandable.
    """

    def __init__(
        self,
        chunks: list[Chunk],
    ):
        self.chunks = chunks

        self.documents = [
            self._tokenize(chunk.text)
            for chunk in chunks
        ]

        self.document_frequency = (
            self._calculate_document_frequency()
        )

        self.total_documents = len(chunks)

    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Convert text into normalized tokens.
        """

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower(),
        )

    # --------------------------------------------------
    # Document frequency
    # --------------------------------------------------

    def _calculate_document_frequency(self):
        """
        Calculate how many documents contain each term.
        """

        frequency = Counter()

        for document in self.documents:

            unique_terms = set(document)

            for term in unique_terms:
                frequency[term] += 1

        return frequency

    # --------------------------------------------------
    # IDF
    # --------------------------------------------------

    def _idf(self, term: str) -> float:
        """
        Calculate inverse document frequency.
        """

        document_frequency = (
            self.document_frequency.get(
                term,
                0,
            )
        )

        if document_frequency == 0:
            return 0.0

        return math.log(
            (self.total_documents + 1)
            / (document_frequency + 1)
        ) + 1

    # --------------------------------------------------
    # TF-IDF score
    # --------------------------------------------------

    def _score_document(
        self,
        query_terms: list[str],
        document_terms: list[str],
    ) -> float:

        if not document_terms:
            return 0.0

        term_frequency = Counter(
            document_terms
        )

        document_length = len(
            document_terms
        )

        score = 0.0

        for term in query_terms:

            if term not in term_frequency:
                continue

            tf = (
                term_frequency[term]
                / document_length
            )

            idf = self._idf(term)

            score += tf * idf

        return score

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_terms = self._tokenize(
            query
        )

        if not query_terms:
            return []

        results = []

        for chunk, document_terms in zip(
            self.chunks,
            self.documents,
        ):

            score = self._score_document(
                query_terms,
                document_terms,
            )

            if score <= 0:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    similarity=score,
                    document_id="",
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                )
            )

        results.sort(
            key=lambda result: result.similarity,
            reverse=True,
        )

        return results[:top_k]