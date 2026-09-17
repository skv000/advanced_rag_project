from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".txt",
}


def is_supported_file(file_path: str | Path) -> bool:
    """
    Check whether a file has a supported extension.
    """
    path = Path(file_path)
    return path.suffix.lower() in SUPPORTED_EXTENSIONS