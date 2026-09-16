from advanced_rag_project.documents.models import Chunk
import re



def split_paragraphs(text: str) -> list[str]:
    """
    Split a document into paragraphs.

    Paragraphs are separated by one or more blank lines.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text.strip(),
    )

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source: str = "unknown",
) -> list[Chunk]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    paragraphs = split_paragraphs(text)

    chunks = []
    current_sentences = []
    current_length = 0

    for paragraph in paragraphs:

        sentences = split_sentences(paragraph)

        for sentence in sentences:

            sentence_length = len(sentence)

            if (
                current_sentences
                and current_length + sentence_length > chunk_size
            ):
                chunks.append(
                    Chunk(
                        text=" ".join(current_sentences).strip(),
                        chunk_id=len(chunks),
                        source=source,
                    )
                )

                # Build sentence-based overlap.
                overlap_sentences = []
                overlap_length = 0

                for previous_sentence in reversed(
                    current_sentences
                ):
                    if (
                        overlap_length
                        + len(previous_sentence)
                        > chunk_overlap
                    ):
                        break

                    overlap_sentences.insert(
                        0,
                        previous_sentence,
                    )

                    overlap_length += len(
                        previous_sentence
                    )

                current_sentences = overlap_sentences

                current_length = sum(
                    len(sentence)
                    for sentence in current_sentences
                )

            current_sentences.append(sentence)
            current_length += sentence_length

    if current_sentences:
        chunks.append(
            Chunk(
                text=" ".join(current_sentences).strip(),
                chunk_id=len(chunks),
                source=source,
            )
        )

    return chunks
