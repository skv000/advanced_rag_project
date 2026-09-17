from pathlib import Path

from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.documents.hashing import calculate_content_hash
from advanced_rag_project.documents.identifiers import create_document_id
from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore


class IngestionPipeline:
    def __init__(
        self,
        persist_directory="data/chroma",
        collection_name="rag_documents",
    ):
        self.embedder = Embedder()

        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def ingest_file(
        self,
        file_path: str,
        chunk_size=500,
        chunk_overlap=50,
    ) -> None:
        """
        Load, hash, check for duplicates, chunk, embed,
        and store a document.
        """

        path = Path(file_path)

        print(f"\nLoading: {path}")

        # --------------------------------------------------
        # 1. Load document
        # --------------------------------------------------

        text = load_text_file(str(path))

        # --------------------------------------------------
        # 2. Create document ID
        # --------------------------------------------------

        document_id = create_document_id(str(path))

        # --------------------------------------------------
        # 3. Calculate content hash
        # --------------------------------------------------

        content_hash = calculate_content_hash(text)

        print(f"Document ID: {document_id}")
        print(f"Content Hash: {content_hash}")

        # --------------------------------------------------
        # 4. Check for duplicate document
        # --------------------------------------------------

        if self.vector_store.document_exists(content_hash):
            print("\nDocument already exists in ChromaDB.")
            print("Skipping ingestion.")
            return

        # --------------------------------------------------
        # 5. Chunk document
        # --------------------------------------------------

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=str(path),
        )

        print(f"Created {len(chunks)} chunks.")

        # --------------------------------------------------
        # 6. Generate embeddings
        # --------------------------------------------------

        embeddings = self.embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )

        # --------------------------------------------------
        # 7. Store in ChromaDB
        # --------------------------------------------------

        print("Adding chunks to ChromaDB...")

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            content_hash=content_hash,
        )

        print("Ingestion complete.")