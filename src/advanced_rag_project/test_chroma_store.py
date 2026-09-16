from advanced_rag_project.documents.models import Chunk
from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.vectorstore.chroma_store import ChromaVectorStore

def main():
    chunks = [
        Chunk(
            text="Python is a programming language.",
            chunk_id=0,
            source="test.txt",
        ),
        Chunk(
            text="RAG combines retrieval with language generation.",
            chunk_id=1,
            source="test.txt",
        ),
        Chunk(
            text="Vector embeddings represent semantic meaning.",
            chunk_id=2,
            source="test.txt",
        ),
    ]

    embedder = Embedder()

    embeddings = embedder.embed_texts(
        [chunk.text for chunk in chunks]
    )

    store = ChromaVectorStore(
        persist_directory="data/chroma_test",
        collection_name="test_collection",
    )

    store.add_chunks(
        chunks=chunks,
        embeddings=embeddings
    )

    query = "How does the retrieval augmented generation work?"

    query_embedding = embedder.embed_text(query)

    results = store.search(
        query_embedding=query_embedding,
        top_k=2,
    )

    print("\nSearch Results:")
    print(results)

if __name__=="__main__":
    main()