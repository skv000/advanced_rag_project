from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.retrieval.retriever import Retriever
from advanced_rag_project.rag.context_builder  import build_context
from advanced_rag_project.rag.prompt_builder import build_rag_prompt
from advanced_rag_project.routing.router import is_rag_question
from advanced_rag_project.llm.ollama_client import (
    SYSTEM_PROMPT,
    stream_chat,
)

def main():

    print("=" * 50)
    print("Advanced RAG Project")
    print("=" * 50)

    # Load Document

    document = load_text_file(
        "data/documents/hindu_philosophy.txt"
    )

    # Create chunks

    chunks = chunk_text(
        text=document,
        chunk_size=200,
        chunk_overlap=30,
        source="hindu_philosophy.txt",
    )

    print(f"Loaded document with {len(chunks)} chunks.")

    # Create retriever

    retriever = Retriever(chunks=chunks)

    # Start Conversation

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    while True:

        question = input("\nYOU : ")

        if question.lower() in {'exit','quit', 'q'}:
            print("\nAI : GoodBye, See you later!")
            break

        use_rag = is_rag_question(question)

        if use_rag:

            results = retriever.search(
                query=question,
                top_k=2,
            )

            context = build_context(
                results=results
            )

            rag_prompt = build_rag_prompt(
                question=question,
                context=context,
            )

            messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            rag_messages = messages[:-1] + [
                {
                    "role": "user",
                    "content": rag_prompt,
                }
            ]

        else:
            messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            rag_messages = messages

        # Generate Response

        print("\nAI : ")

        answer_parts = []

        for chunk in stream_chat(
            messages=rag_messages
        ):
            print(
                chunk,
                flush=True,
                end=""
            )

            answer_parts.append(chunk)

        print()

        # Save assistant response

        answer = "".join(answer_parts)

        messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


if __name__=="__main__":
    main()
