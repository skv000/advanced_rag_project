import hashlib


def calculate_content_hash(text: str) -> str:
    """
    Calculate a stable SHA-256 hash for document content.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()