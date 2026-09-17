from pathlib import Path

from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.documents.file_types import (
    is_supported_file,
)
from advanced_rag_project.documents.identifiers import (
    create_document_id,
)
from advanced_rag_project.documents.loader import (
    load_document,
)
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.ingestion.results import (
    IngestionStats,
)
from advanced_rag_project.vectorstore.chroma_store import (
    ChromaVectorStore,
)


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
    ) -> str:
        """
        Load, validate, hash, detect document state,
        chunk, embed, and store a document.

        Returns:

            "new"
                Document was newly ingested.

            "unchanged"
                Document already exists and content
                has not changed.

            "modified"
                Document was modified and replaced.

            "duplicate"
                Document content already exists under
                another document.
        """

        path = Path(file_path)

        print(f"\nLoading: {path}")

        # --------------------------------------------------
        # Validate file type
        # --------------------------------------------------

        if not is_supported_file(path):
            raise ValueError(
                f"Unsupported file type: {path.suffix}"
            )

        # --------------------------------------------------
        # Load document
        # --------------------------------------------------

        document = load_document(path)

        document_id = document.document_id
        content_hash = document.content_hash

        print(
            f"Document ID: {document_id}"
        )

        print(
            f"Content Hash: {content_hash}"
        )

        print(
            f"File Type: {document.file_type}"
        )

        print(
            f"File Size: {document.file_size} bytes"
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
            # Check duplicate content
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

                return "duplicate"

            document_status = "new"

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

            return "unchanged"

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

            document_status = "modified"

        # --------------------------------------------------
        # Chunk document
        # --------------------------------------------------

        chunks = chunk_text(
            document.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=document.source,
        )

        if not chunks:
            raise ValueError(
                f"No chunks were created for document: "
                f"{path}"
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

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Embedding count does not match "
                "chunk count."
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
            file_type=document.file_type,
            file_size=document.file_size,
            modified_time=document.modified_time,
        )

        print(
            "Ingestion complete."
        )

        return document_status

    def ingest_directory(
        self,
        directory: str,
        chunk_size=500,
        chunk_overlap=50,
    ) -> IngestionStats:
        """
        Incrementally synchronize supported documents
        in a directory and all of its subdirectories
        with ChromaDB.

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

        DUPLICATE:
            Document contains content already stored
            under another document.

        FAILED:
            Document could not be loaded or ingested.

        Unsupported files are ignored.
        """

        directory_path = Path(directory)

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
        # Find all files recursively
        # --------------------------------------------------

        all_files = sorted(
            path
            for path in directory_path.rglob("*")
            if path.is_file()
        )

        # --------------------------------------------------
        # Filter supported files
        # --------------------------------------------------

        files = [
            file_path
            for file_path in all_files
            if is_supported_file(file_path)
        ]

        # --------------------------------------------------
        # Find unsupported files
        # --------------------------------------------------

        ignored_files = [
            file_path
            for file_path in all_files
            if not is_supported_file(file_path)
        ]

        stats = IngestionStats(
            scanned=len(files)
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
            f"Supported documents found: "
            f"{len(files)}"
        )

        print(
            f"Unsupported files ignored: "
            f"{len(ignored_files)}"
        )

        if ignored_files:

            print(
                "\nIgnored files:"
            )

            for file_path in ignored_files:

                print(
                    f"  - "
                    f"{file_path.relative_to(directory_path)}"
                )

        # --------------------------------------------------
        # Create document IDs for supported files
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

                stats.deleted += 1

        else:

            print(
                "\nNo deleted documents detected."
            )

        # --------------------------------------------------
        # Process supported documents
        # --------------------------------------------------

        if not files:

            print(
                "\nNo supported documents found."
            )

        for file_path in files:

            try:

                status = self.ingest_file(
                    file_path=str(file_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                if status == "new":

                    stats.new += 1

                elif status == "unchanged":

                    stats.unchanged += 1

                elif status == "modified":

                    stats.modified += 1

                elif status == "duplicate":

                    stats.duplicates += 1

                else:

                    print(
                        f"Unknown ingestion status: "
                        f"{status}"
                    )

                    stats.failed += 1

            except Exception as exc:

                stats.failed += 1

                print(
                    f"\nFailed to ingest "
                    f"{file_path}: {exc}"
                )

        # --------------------------------------------------
        # Finish
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

        print(
            f"Scanned:      {stats.scanned}"
        )

        print(
            f"New:          {stats.new}"
        )

        print(
            f"Unchanged:    {stats.unchanged}"
        )

        print(
            f"Modified:     {stats.modified}"
        )

        print(
            f"Deleted:      {stats.deleted}"
        )

        print(
            f"Duplicates:   {stats.duplicates}"
        )

        print(
            f"Failed:       {stats.failed}"
        )

        return stats