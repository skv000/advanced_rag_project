from ollama import chat

MODEL_NAME = "llama3.2:3b"

SYSTEM_PROMPT = """
You are funny, charismatic and knowledgeble assistant.
you can answer questions in funny way.   
"""

# First version - for testing. No streaming and full response after complete generataion
def ask_llm(prompt: str) -> str:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role":"user",
                "content":prompt,
            }
        ]
    )

    return response

# This fuction defined for chat_history feature only
def chat_with_llm(messages: list[dict]) -> str:
    response = chat(
        model=MODEL_NAME,
        messages=messages
    )

    return response.message.content


def stream_chat(messages: list[dict]):
    stream = chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        content = chunk.message.content
        # print(chunk)

        if content:
            yield content
