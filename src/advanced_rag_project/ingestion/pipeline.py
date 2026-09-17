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
        Load, hash, detect document state, and ingest
        the document.

        Document states:

        NEW:
            Document does not exist.

        UNCHANGED:
            Document exists and content hash matches.

        MODIFIED:
            Document exists but content hash changed.

        DUPLICATE:
            Another document already contains the
            same content.
        """

        path = Path(file_path)

        print(f"\nLoading: {path}")

        # --------------------------------------------------
        # Load document
        # --------------------------------------------------

        text = load_text_file(
            str(path)
        )

        # --------------------------------------------------
        # Create stable document ID
        # --------------------------------------------------

        document_id = create_document_id(
            str(path)
        )

        # --------------------------------------------------
        # Calculate content hash
        # --------------------------------------------------

        content_hash = calculate_content_hash(
            text
        )

        print(
            f"Document ID: {document_id}"
        )

        print(
            f"Content Hash: {content_hash}"
        )

        # --------------------------------------------------
        # Check existing document
        # --------------------------------------------------

        stored_hash = (
            self.vector_store.get_document_hash(
                document_id
            )
        )

        # --------------------------------------------------
        # NEW DOCUMENT
        # --------------------------------------------------

        if stored_hash is None:

            print(
                "\nDocument status: NEW"
            )

            # ----------------------------------------------
            # Check for duplicate content
            # ----------------------------------------------

            if self.vector_store.document_exists(
                content_hash
            ):

                print(
                    "Duplicate content detected."
                )

                print(
                    "Skipping ingestion."
                )

                return

        # --------------------------------------------------
        # UNCHANGED DOCUMENT
        # --------------------------------------------------

        elif stored_hash == content_hash:

            print(
                "\nDocument status: UNCHANGED"
            )

            print(
                "Document already exists in ChromaDB."
            )

            print(
                "Skipping ingestion."
            )

            return

        # --------------------------------------------------
        # MODIFIED DOCUMENT
        # --------------------------------------------------

        else:

            print(
                "\nDocument status: MODIFIED"
            )

            print(
                "Removing previous document chunks..."
            )

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

        print(
            f"Created {len(chunks)} chunks."
        )

        # --------------------------------------------------
        # Generate embeddings
        # --------------------------------------------------

        embeddings = self.embedder.embed_texts(
            [
                chunk.text
                for chunk in chunks
            ]
        )

        # --------------------------------------------------
        # Store in ChromaDB
        # --------------------------------------------------

        print(
            "Adding chunks to ChromaDB..."
        )

        self.vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id=document_id,
            content_hash=content_hash,
        )

        print(
            "Ingestion complete."
        )

    def ingest_directory(
        self,
        directory: str,
        chunk_size=500,
        chunk_overlap=50,
    ) -> None:
        """
        Incrementally synchronize all .txt documents
        in a directory with ChromaDB.

        Handles:

        NEW:
            Document exists on disk but not in ChromaDB.

        UNCHANGED:
            Document exists in both places and content
            is unchanged.

        MODIFIED:
            Document exists in both places but content
            changed.

        DELETED:
            Document exists in ChromaDB but no longer
            exists in the filesystem.
        """

        directory_path = Path(
            directory
        )

        # --------------------------------------------------
        # Validate directory
        # --------------------------------------------------

        if not directory_path.exists():

            raise FileNotFoundError(
                f"Directory does not exist: "
                f"{directory_path}"
            )

        if not directory_path.is_dir():

            raise NotADirectoryError(
                f"Path is not a directory: "
                f"{directory_path}"
            )

        # --------------------------------------------------
        # Find .txt files
        # --------------------------------------------------

        files = sorted(
            directory_path.glob("*.txt")
        )

        print(
            "=" * 60
        )

        print(
            "DIRECTORY INGESTION"
        )

        print(
            "=" * 60
        )

        print(
            f"Directory: {directory_path}"
        )

        print(
            f"Documents found: {len(files)}"
        )

        # --------------------------------------------------
        # Create document IDs for files on disk
        # --------------------------------------------------

        filesystem_document_ids = {
            create_document_id(
                str(file_path)
            )
            for file_path in files
        }

        # --------------------------------------------------
        # Get document IDs stored in ChromaDB
        # --------------------------------------------------

        stored_document_ids = (
            self.vector_store.get_document_ids()
        )

        # --------------------------------------------------
        # Detect deleted documents
        #
        # Stored IDs - Filesystem IDs
        #
        # Anything remaining was deleted from disk.
        # --------------------------------------------------

        deleted_document_ids = (
            stored_document_ids
            - filesystem_document_ids
        )

        if deleted_document_ids:

            print(
                "\nDeleted documents detected:"
            )

            for document_id in sorted(
                deleted_document_ids
            ):

                print(
                    f"  - {document_id}"
                )

                self.vector_store.delete_document(
                    document_id=document_id
                )

                print(
                    "    Removed from ChromaDB."
                )

        # --------------------------------------------------
        # No deleted documents
        # --------------------------------------------------

        else:

            print(
                "\nNo deleted documents detected."
            )

        # --------------------------------------------------
        # Process documents currently on disk
        # --------------------------------------------------

        if not files:

            print(
                "\nNo .txt documents found."
            )

        for file_path in files:

            self.ingest_file(
                file_path=str(file_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        # --------------------------------------------------
        # Finished
        # --------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "DIRECTORY INGESTION COMPLETE"
        )

        print(
            "=" * 60
        )