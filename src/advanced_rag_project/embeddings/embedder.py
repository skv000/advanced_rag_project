from sentence_transformers import SentenceTransformer
import torch


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Embedding device: {device}")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
        )

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text)

        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)

        return [
            embedding.tolist()
            for embedding in embeddings
        ]
