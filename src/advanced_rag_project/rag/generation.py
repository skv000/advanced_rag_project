from dataclasses import dataclass

from advanced_rag_project.rag.context_engineering import ContextResult


@dataclass
class GenerationResult:
    """
    Structured result produced after RAG generation.

    This object keeps the generated answer together with
    grounding and source diagnostics.
    """

    answer: str
    grounded: bool
    insufficient_context: bool
    sources: list[str]
    context_items: int
    estimated_context_tokens: int
    unsupported_sentences: list[str]


class AnswerValidator:
    """
    Deterministic answer validator.

    The validator does not use another LLM.

    It checks whether generated answer sentences have
    meaningful lexical overlap with the supplied context.

    This is intentionally a lightweight safety/diagnostic
    mechanism, not a true semantic faithfulness evaluator.
    """

    def __init__(
        self,
        minimum_sentence_overlap: float = 0.15,
    ):
        if not 0 <= minimum_sentence_overlap <= 1:
            raise ValueError(
                "minimum_sentence_overlap must be between 0 and 1"
            )

        self.minimum_sentence_overlap = (
            minimum_sentence_overlap
        )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """
        Convert text into a normalized set of tokens.

        Stopwords are intentionally not removed at this stage.
        """

        import re

        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split generated text into simple sentences.
        """

        import re

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def validate(
        self,
        answer: str,
        context_result: ContextResult,
    ) -> tuple[
        bool,
        bool,
        list[str],
    ]:
        """
        Validate a generated answer against retrieved context.

        Returns:

            grounded
            insufficient_context
            unsupported_sentences
        """

        answer = answer.strip()

        if not context_result.items:
            return (
                False,
                True,
                self._split_sentences(answer),
            )

        if not answer:
            return (
                False,
                False,
                [],
            )

        context_tokens = self._tokenize(
            context_result.context
        )

        unsupported_sentences = []

        for sentence in self._split_sentences(answer):

            sentence_tokens = self._tokenize(
                sentence
            )

            if not sentence_tokens:
                continue

            overlap = (
                len(sentence_tokens & context_tokens)
                / len(sentence_tokens)
            )

            if (
                overlap
                < self.minimum_sentence_overlap
            ):
                unsupported_sentences.append(
                    sentence
                )

        grounded = (
            len(unsupported_sentences) == 0
        )

        return (
            grounded,
            False,
            unsupported_sentences,
        )


def extract_sources(
    context_result: ContextResult,
) -> list[str]:
    """
    Extract unique source names from the final
    context in relevance order.
    """

    sources = []
    seen = set()

    for item in context_result.items:

        source = item.source.strip()

        if not source:
            continue

        if source in seen:
            continue

        seen.add(source)
        sources.append(source)

    return sources


def create_generation_result(
    answer: str,
    context_result: ContextResult,
    validator: AnswerValidator | None = None,
) -> GenerationResult:
    """
    Create a structured GenerationResult from a generated
    answer and ContextResult.
    """

    if validator is None:
        validator = AnswerValidator()

    (
        grounded,
        insufficient_context,
        unsupported_sentences,
    ) = validator.validate(
        answer=answer,
        context_result=context_result,
    )

    return GenerationResult(
        answer=answer,
        grounded=grounded,
        insufficient_context=insufficient_context,
        sources=extract_sources(
            context_result
        ),
        context_items=len(
            context_result.items
        ),
        estimated_context_tokens=(
            context_result.estimated_tokens
        ),
        unsupported_sentences=(
            unsupported_sentences
        ),
    )