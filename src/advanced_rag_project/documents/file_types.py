from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


def is_supported_file(file_path: str | Path) -> bool:
    """
    Return True if the file extension is supported
    by the ingestion pipeline.
    """

    path = Path(file_path)

    return path.suffix.lower() in SUPPORTED_EXTENSIONS