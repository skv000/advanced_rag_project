import chromadb

from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import (
    ChromaVectorStore,
)


def main():
    chunks_a = [
        Chunk(
            text="Document A first chunk.",
            chunk_id=0,
            source="document_a.txt",
        ),
        Chunk(
            text="Document A second chunk.",
            chunk_id=1,
            source="document_a.txt",
        ),
    ]

    chunks_b = [
        Chunk(
            text="Document B first chunk.",
            chunk_id=0,
            source="document_b.txt",
        ),
        Chunk(
            text="Document B second chunk.",
            chunk_id=1,
            source="document_b.txt",
        ),
    ]

    embedder = Embedder()

    embeddings_a = embedder.embed_texts(
        [chunk.text for chunk in chunks_a]
    )

    embeddings_b = embedder.embed_texts(
        [chunk.text for chunk in chunks_b]
    )

    store = ChromaVectorStore(
        persist_directory="data/chroma_collision_test",
        collection_name="documents",
    )

    store.add_chunks(
        chunks=chunks_a,
        embeddings=embeddings_a,
        document_id="document_a",
    )

    store.add_chunks(
        chunks=chunks_b,
        embeddings=embeddings_b,
        document_id="document_b",
    )

    client = chromadb.PersistentClient(
        path="data/chroma_collision_test"
    )

    collection = client.get_collection("documents")

    print("\nTotal records:", collection.count())

    result = collection.get(
        include=["documents", "metadatas"]
    )

    print("\nIDs:")
    print(result["ids"])

    print("\nMetadata:")

    for metadata in result["metadatas"]:
        print(metadata)


if __name__ == "__main__":
    main()