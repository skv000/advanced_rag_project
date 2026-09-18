from collections.abc import Iterator

from ollama import chat


MODEL_NAME = "llama3.2:3b"


SYSTEM_PROMPT = """
You are a funny, charismatic and knowledgeable assistant.

When answering RAG questions, strictly follow
the context supplied by the user prompt.
"""


def ask_llm(prompt: str) -> str:
    """
    Generate a complete response from Ollama.
    """

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.message.content


def chat_with_llm(
    messages: list[dict],
) -> str:
    """
    Generate a complete response using chat history.
    """

    response = chat(
        model=MODEL_NAME,
        messages=messages,
    )

    return response.message.content


def stream_chat(
    messages: list[dict],
) -> Iterator[str]:
    """
    Stream generated content from Ollama.
    """

    stream = chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )

    for chunk in stream:

        content = chunk.message.content

        if content:
            yield content