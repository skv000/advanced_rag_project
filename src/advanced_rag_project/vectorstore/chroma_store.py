import chromadb

from advanced_rag_project.documents.models import Chunk


class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: str = "data/chroma",
        collection_name: str = "rag_documents",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def add_chunks(
        self,
        chunks,
        embeddings,
        document_id: str,
        content_hash: str,
        file_type: str | None = None,
        file_size: int | None = None,
        modified_time: float | None = None,
    ):
        """
        Add document chunks and metadata to ChromaDB.
        """

        ids = [
            f"{document_id}:{chunk.chunk_id}"
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            {
                "document_id": document_id,
                "content_hash": content_hash,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "file_type": file_type or "",
                "file_size": file_size or 0,
                "modified_time": modified_time or 0.0,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding,
        top_k=3,
        filters: dict | None = None,
    ):
        """
        Search the vector store.
        """

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
        )

    def document_exists(
        self,
        content_hash: str,
    ) -> bool:
        """
        Check whether a document with the given
        content hash already exists in ChromaDB.
        """

        results = self.collection.get(
            where={
                "content_hash": content_hash
            },
            limit=1,
        )

        return len(results["ids"]) > 0

    def get_document_hash(
        self,
        document_id: str,
    ) -> str | None:
        """
        Get the stored content hash for a document.
        """

        results = self.collection.get(
            where={
                "document_id": document_id
            },
            limit=1,
            include=["metadatas"],
        )

        if not results["metadatas"]:
            return None

        return results["metadatas"][0][
            "content_hash"
        ]

    def get_document_ids(self) -> set[str]:
        """
        Get all unique document IDs currently stored
        in ChromaDB.
        """

        results = self.collection.get(
            include=["metadatas"]
        )

        document_ids = {
            metadata["document_id"]
            for metadata in results["metadatas"]
            if metadata
            and "document_id" in metadata
        }

        return document_ids

    def get_chunks(self) -> list[Chunk]:
        """
        Load all stored chunks from ChromaDB.

        ChromaDB stores the chunk text as `documents`
        and chunk metadata separately. This method
        reconstructs the application's Chunk objects
        from those persisted values.
        """

        results = self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        documents = results.get(
            "documents",
            [],
        )

        metadatas = results.get(
            "metadatas",
            [],
        )

        chunks = []

        for document, metadata in zip(
            documents,
            metadatas,
        ):
            if not document or not metadata:
                continue

            chunks.append(
                Chunk(
                    text=document,
                    chunk_id=int(
                        metadata["chunk_id"]
                    ),
                    source=metadata["source"],
                )
            )

        chunks.sort(
            key=lambda chunk: (
                chunk.source,
                chunk.chunk_id,
            )
        )

        return chunks

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete all chunks belonging to a document.
        """

        self.collection.delete(
            where={
                "document_id": document_id
            }
        )