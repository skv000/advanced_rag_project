from advanced_rag_project.llm.ollama_client import (
    SYSTEM_PROMPT,
    stream_chat,
)
from advanced_rag_project.rag.context_builder import (
    build_context_result,
)
from advanced_rag_project.rag.generation import (
    AnswerValidator,
    GenerationResult,
    create_generation_result,
)
from advanced_rag_project.rag.prompt_builder import (
    build_rag_prompt,
)
from advanced_rag_project.retrieval.reranking_retriever import (
    RerankingRetriever,
)
from advanced_rag_project.routing.router import (
    is_rag_question,
)


def generate_rag_answer(
    question: str,
    retriever: RerankingRetriever,
    validator: AnswerValidator,
) -> GenerationResult:
    """
    Execute the complete grounded RAG generation pipeline.
    """

    results = retriever.search(
        query=question,
        candidate_k=6,
        top_k=3,
        dense_weight=0.7,
        keyword_weight=0.3,
    )

    context_result = build_context_result(
        results=results,
        max_context_tokens=1500,
    )

    prompt = build_rag_prompt(
        question=question,
        context=context_result.context,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    print("\nAI : ")

    answer_parts = []

    for chunk in stream_chat(
        messages=messages
    ):
        print(
            chunk,
            flush=True,
            end="",
        )

        answer_parts.append(chunk)

    print()

    answer = "".join(
        answer_parts
    ).strip()

    return create_generation_result(
        answer=answer,
        context_result=context_result,
        validator=validator,
    )


def main():

    print("=" * 60)
    print("Advanced RAG Project")
    print("Phase 11 - Grounded Generation")
    print("=" * 60)

    retriever = RerankingRetriever(
        persist_directory="data/chroma",
        collection_name="rag_documents",
    )

    print(
        "Connected to Hybrid Retrieval + "
        "Cross-Encoder Reranking."
    )

    validator = AnswerValidator(
        minimum_sentence_overlap=0.15,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    while True:

        question = input(
            "\nYOU : "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print(
                "\nAI : Goodbye, see you later!"
            )
            break

        if not question:
            continue

        use_rag = is_rag_question(
            question
        )

        if use_rag:

            generation_result = (
                generate_rag_answer(
                    question=question,
                    retriever=retriever,
                    validator=validator,
                )
            )

            messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        generation_result.answer
                    ),
                }
            )

            print(
                "\n"
                "Grounded      : "
                f"{generation_result.grounded}"
            )

            print(
                "Insufficient   : "
                f"{generation_result.insufficient_context}"
            )

            print(
                "Context chunks: "
                f"{generation_result.context_items}"
            )

            print(
                "Context tokens: "
                f"{generation_result.estimated_context_tokens}"
            )

            if generation_result.sources:

                print(
                    "Sources:"
                )

                for source in (
                    generation_result.sources
                ):
                    print(
                        f"  - {source}"
                    )

            if (
                generation_result
                .unsupported_sentences
            ):

                print(
                    "Potentially unsupported "
                    "sentences:"
                )

                for sentence in (
                    generation_result
                    .unsupported_sentences
                ):
                    print(
                        f"  - {sentence}"
                    )

        else:

            messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            print("\nAI : ")

            answer_parts = []

            for chunk in stream_chat(
                messages=messages
            ):
                print(
                    chunk,
                    flush=True,
                    end="",
                )

                answer_parts.append(
                    chunk
                )

            print()

            answer = "".join(
                answer_parts
            ).strip()

            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


if __name__ == "__main__":
    main()