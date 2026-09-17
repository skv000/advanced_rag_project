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

        DUPLICATE:
            Another document already contains the same content.
        """

        path = Path(file_path)

        print(f"\nLoading: {path}")

        text = load_text_file(str(path))

        document_id = create_document_id(str(path))
        content_hash = calculate_content_hash(text)

        print(f"Document ID: {document_id}")
        print(f"Content Hash: {content_hash}")

        # --------------------------------------------------
        # Check whether this exact document already exists
        # --------------------------------------------------

        stored_hash = self.vector_store.get_document_hash(
            document_id
        )

        if stored_hash is None:
            print("\nDocument status: NEW")

            # ----------------------------------------------
            # Check whether another document has same content
            # ----------------------------------------------

            if self.vector_store.document_exists(
                content_hash
            ):
                print("Duplicate content detected.")
                print("Skipping ingestion.")
                return

        elif stored_hash == content_hash:
            print("\nDocument status: UNCHANGED")
            print("Document already exists in ChromaDB.")
            print("Skipping ingestion.")
            return

        else:
            print("\nDocument status: MODIFIED")
            print("Removing previous document chunks...")

            self.vector_store.delete_document(
                document_id=document_id
            )

        # --------------------------------------------------
        # Chunk document
        # --------------------------------------------------

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=str(path),
        )

        print(f"Created {len(chunks)} chunks.")

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        embeddings = self.embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )

        # --------------------------------------------------
        # Store in ChromaDB
        # --------------------------------------------------

        print("Adding chunks to ChromaDB...")

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            content_hash=content_hash,
        )

        print("Ingestion complete.")

    def ingest_directory(
        self,
        directory: str,
        chunk_size=500,
        chunk_overlap=50,
    ) -> None:
        """
        Incrementally ingest all .txt documents in a directory.
        """

        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory_path}"
            )

        if not directory_path.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {directory_path}"
            )

        files = sorted(
            directory_path.glob("*.txt")
        )

        print("=" * 60)
        print("DIRECTORY INGESTION")
        print("=" * 60)

        print(f"Directory: {directory_path}")
        print(f"Documents found: {len(files)}")

        if not files:
            print("\nNo .txt documents found.")
            return

        for file_path in files:
            self.ingest_file(
                file_path=str(file_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        print("\n" + "=" * 60)
        print("DIRECTORY INGESTION COMPLETE")
        print("=" * 60)