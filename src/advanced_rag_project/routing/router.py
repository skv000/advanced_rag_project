def is_rag_question(question: str) -> bool:
    """
    Decide whether a question should use document retrieval.
    """

    question_lower = question.lower().strip()

    chat_questions = {
        "what is my name",
        "what's my name",
        "who am i",
        "what did i say",
        "what did you say",
        "who are you",
        "hello",
        "hi",
    }

    if question_lower in chat_questions:
        return False

    rag_keywords = {
        "advaita",
        "vedanta",
        "samkhya",
        "yoga",
        "bhakti",
        "philosophy",
        "dharma",
        "karma",
        "brahman",
        "atman",
        "purusha",
        "prakriti",
    }

    return any(
        keyword in question_lower
        for keyword in rag_keywords
    )
