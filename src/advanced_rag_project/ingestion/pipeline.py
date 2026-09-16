from pathlib import Path

from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


class IngestionPipeline:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_documents",
    ):
        self.embedder = Embedder()

        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:

        path = Path(file_path)

        print(f"\nLoading: {path}")

        text = load_text_file(str(path))

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=str(path),
        )

        print(f"Created {len(chunks)} chunks.")

        embeddings = self.embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )

        print("Adding chunks to ChromaDB...")

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        print("Ingestion complete.")