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
        Load, hash, detect document state, and ingest the document.

        Document states:

        NEW:
            Document does not exist.

        UNCHANGED:
            Document exists and content hash matches.

        MODIFIED:
            Document exists but content hash changed.
        """

        path = Path(file_path)

        print(f"\nLoading: {path}")

        # --------------------------------------------------
        # 1. Load document
        # --------------------------------------------------

        text = load_text_file(str(path))

        # --------------------------------------------------
        # 2. Identify document
        # --------------------------------------------------

        document_id = create_document_id(str(path))

        # --------------------------------------------------
        # 3. Calculate content hash
        # --------------------------------------------------

        content_hash = calculate_content_hash(text)

        print(f"Document ID: {document_id}")
        print(f"Content Hash: {content_hash}")

        # --------------------------------------------------
        # 4. Check existing document
        # --------------------------------------------------

        stored_hash = self.vector_store.get_document_hash(
            document_id
        )

        # --------------------------------------------------
        # 5. NEW document
        # --------------------------------------------------

        if stored_hash is None:
            print("\nDocument status: NEW")

        # --------------------------------------------------
        # 6. UNCHANGED document
        # --------------------------------------------------

        elif stored_hash == content_hash:
            print("\nDocument status: UNCHANGED")
            print("Document already exists in ChromaDB.")
            print("Skipping ingestion.")
            return

        # --------------------------------------------------
        # 7. MODIFIED document
        # --------------------------------------------------

        else:
            print("\nDocument status: MODIFIED")
            print("Removing previous document chunks...")

            self.vector_store.delete_document(
                document_id=document_id
            )

        # --------------------------------------------------
        # 8. Chunk document
        # --------------------------------------------------

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=str(path),
        )

        print(f"Created {len(chunks)} chunks.")

        # --------------------------------------------------
        # 9. Generate embeddings
        # --------------------------------------------------

        embeddings = self.embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )

        # --------------------------------------------------
        # 10. Store document
        # --------------------------------------------------

        print("Adding chunks to ChromaDB...")

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            content_hash=content_hash,
        )

        print("Ingestion complete.")