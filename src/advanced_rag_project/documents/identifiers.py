from pathlib import Path


def create_document_id(file_path: str) -> str:
    """
    Create a stable document ID from a file path.
    """

    path = Path(file_path)

    return path.stem.lower().replace(" ", "_")