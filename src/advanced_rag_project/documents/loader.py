from pathlib import Path

def load_text_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileExistsError(f"file not found: {file_path}")

    if path.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported")

    return path.read_text(encoding="utf-8")
