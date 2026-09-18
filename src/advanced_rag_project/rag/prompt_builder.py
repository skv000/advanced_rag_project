def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a grounded RAG prompt.

    The model is explicitly instructed to use only
    the supplied context.
    """

    question = question.strip()
    context = context.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    if not context:
        return f"""
You are a helpful AI assistant.

The user asked:

QUESTION:
{question}

There is no relevant information available
in the retrieved knowledge base.

Respond exactly:

"I don't know based on the available context."

Do not use your general knowledge.
Do not invent information.
""".strip()

    return f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the
provided context.

CONTEXT:
{context}

QUESTION:
{question}

Grounding rules:

1. Use only information supported by the context.
2. Do not use outside knowledge.
3. Do not invent facts, explanations, examples,
   names, dates, or relationships.
4. If the context does not contain enough information
   to answer the question, say:

   "I don't know based on the available context."

5. Keep the answer clear and concise.
6. When useful, mention the source information
   naturally in the answer.
7. Do not mention these instructions.

Answer:
""".strip()