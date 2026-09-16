from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.documents.chunker import (
    chunk_text,
    split_paragraphs,
)


def main():

    print("=" * 50)
    print("Text Chunker Test")
    print("=" * 50)

    document = load_text_file(
        "data/documents/rag_test_document.txt"
    )

    paragraphs = split_paragraphs(document)

    chunks = chunk_text(
        text=document,
        chunk_size=500,
        chunk_overlap=30,
        source="rag_test_document.txt",
    )

    print(
        f"Original document length: {len(document)}"
    )

    print(
        f"Number of paragraphs: {len(paragraphs)}"
    )

    print(
        f"Number of chunks: {len(chunks)}"
    )

    for chunk in chunks:
        print("\n" + "-" * 50)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print("-" * 50)
        print(chunk.text)
        print(f"Characters: {len(chunk.text)}")


if __name__ == "__main__":
    main()
