from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.rag.context_builder import (
    build_context_result,
)
from advanced_rag_project.rag.generation import (
    AnswerValidator,
    create_generation_result,
)
from advanced_rag_project.rag.prompt_builder import (
    build_rag_prompt,
)


def main():

    print("=" * 60)
    print("PHASE 11 - GROUNDED GENERATION TEST")
    print("=" * 60)

    chunks = [
        Chunk(
            text=(
                "Advaita Vedanta teaches that reality "
                "is fundamentally non-dual."
            ),
            chunk_id=0,
            source="rag_test_document.txt",
        ),
        Chunk(
            text=(
                "The tradition distinguishes between "
                "the apparent individual self and "
                "ultimate reality."
            ),
            chunk_id=1,
            source="rag_test_document.txt",
        ),
        Chunk(
            text=(
                "Samkhya describes reality using "
                "Purusha and Prakriti."
            ),
            chunk_id=2,
            source="philosophy.txt",
        ),
    ]

    class FakeRetrievalResult:

        def __init__(
            self,
            chunk,
            similarity,
        ):
            self.chunk = chunk
            self.similarity = similarity
            self.document_id = ""
            self.source = chunk.source
            self.chunk_id = chunk.chunk_id

    results = [
        FakeRetrievalResult(
            chunks[0],
            0.95,
        ),
        FakeRetrievalResult(
            chunks[1],
            0.80,
        ),
        FakeRetrievalResult(
            chunks[2],
            0.40,
        ),
    ]

    # ---------------------------------------------------------
    # Context engineering
    # ---------------------------------------------------------

    context_result = build_context_result(
        results=results,
        max_context_tokens=500,
    )

    assert context_result.items

    assert (
        "Advaita Vedanta"
        in context_result.context
    )

    print(
        "Context items:",
        len(context_result.items),
    )

    print(
        "Estimated tokens:",
        context_result.estimated_tokens,
    )

    # ---------------------------------------------------------
    # Prompt construction
    # ---------------------------------------------------------

    prompt = build_rag_prompt(
        question=(
            "What does Advaita Vedanta teach?"
        ),
        context=context_result.context,
    )

    assert (
        "ONLY the" in prompt
    )

    assert (
        "Do not use outside knowledge"
        in prompt
    )

    print(
        "Grounded prompt created."
    )

    # ---------------------------------------------------------
    # Supported answer
    # ---------------------------------------------------------

    supported_answer = (
        "Advaita Vedanta teaches that "
        "reality is fundamentally non-dual."
    )

    validator = AnswerValidator(
        minimum_sentence_overlap=0.15,
    )

    result = create_generation_result(
        answer=supported_answer,
        context_result=context_result,
        validator=validator,
    )

    assert result.grounded is True

    assert result.insufficient_context is False

    assert (
        result.unsupported_sentences == []
    )

    assert (
        "rag_test_document.txt"
        in result.sources
    )

    print(
        "Supported answer validation: PASSED"
    )

    # ---------------------------------------------------------
    # Unsupported answer
    # ---------------------------------------------------------

    unsupported_answer = (
        "Advaita Vedanta teaches non-duality. "
        "It was founded by a philosopher who "
        "lived exactly 1500 years ago."
    )

    result = create_generation_result(
        answer=unsupported_answer,
        context_result=context_result,
        validator=validator,
    )

    assert result.grounded is False

    assert result.insufficient_context is False

    assert result.unsupported_sentences

    print(
        "Unsupported answer detection: PASSED"
    )

    # ---------------------------------------------------------
    # Empty context
    # ---------------------------------------------------------

    empty_context = build_context_result(
        results=[],
    )

    result = create_generation_result(
        answer=(
            "I don't know based on the "
            "available context."
        ),
        context_result=empty_context,
        validator=validator,
    )

    assert (
        result.insufficient_context is True
    )

    assert result.grounded is False

    print(
        "Insufficient context detection: PASSED"
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print(
        "Sources:",
        result.sources,
    )

    print(
        "PHASE 11 GROUNDED GENERATION "
        "TEST PASSED"
    )


if __name__ == "__main__":
    main()