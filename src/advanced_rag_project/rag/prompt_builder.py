def build_rag_prompt(
        question: str,
        context: str,
) -> str:
    """
    Build a prompt that instruct the LLm to answer using the retrieved context.
    """

    prompt = f"""
You are helpful AI assistant.

Answer the user's questtion using the provided context below.

CONTEXT:
{context}

QUESTION:
{question}

Instructions:
- Use the provided context to answer the question.
- If the answer is not available in the context, say you don't know.
- Do not makeup information.
- Give a clear and concise answer.
"""

    return prompt.strip()
