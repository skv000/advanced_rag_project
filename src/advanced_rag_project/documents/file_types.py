from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
}


def is_supported_file(file_path: str | Path) -> bool:
    """
    Return True when the file extension is supported.
    """

    path = Path(file_path)

    return path.suffix.lower() in SUPPORTED_EXTENSIONS