from pathlib import Path

from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.documents.identifiers import create_document_id
from advanced_rag_project.documents.models import Document


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


def _read_text_file(file_path: Path) -> str:
    """
    Read a text-based document with UTF-8 encoding.

    errors="replace" prevents a single invalid byte from
    crashing the entire ingestion batch.
    """
    return file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def _normalize_text(text: str) -> str:
    """
    Normalize document text before chunking.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove trailing whitespace from each line
    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # Remove excessive blank lines
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def load_document(file_path: str | Path) -> Document:
    """
    Load a supported TXT or Markdown document.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Path is not a file: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    raw_text = _read_text_file(path)
    text = _normalize_text(raw_text)

    if not text:
        raise ValueError(
            f"Document is empty: {path}"
        )

    content_hash = calculate_content_hash(text)
    document_id = create_document_id(str(path))

    stat = path.stat()

    return Document(
        text=text,
        source=str(path),
        document_id=document_id,
        content_hash=content_hash,
        file_type=extension,
        file_size=stat.st_size,
        modified_time=stat.st_mtime,
    )