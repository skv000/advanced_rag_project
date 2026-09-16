from advanced_rag_project.documents.models import Chunk


def build_context(
    results: list[tuple[Chunk, float]]
) -> str:
    """
    Combine retrieved chunks into a single context string.

    Each chunk includes metadata such as chunk ID
    and source document.
    """

    context_parts = []

    for chunk, score in results:
        context_parts.append(
            f"[Source: {chunk.source} | "
            f"Chunk ID: {chunk.chunk_id} | "
            f"Similarity: {score:.4f}]\n"
            f"{chunk.text}"
        )

    return "\n\n".join(context_parts)
