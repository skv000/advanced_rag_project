from advanced_rag_project.embeddings.embedder import Embedder
from advanced_rag_project.retrieval.similarity import cosine_similarity


def main():
    print("=" * 50)
    print("Semantic Similarity Test")
    print("=" * 50)

    embedder = Embedder()

    texts = [
        "Advaita Vedanta teaches non-duality.",
        "Advaita Vedanta describes reality as non-dual.",
        "The weather is cloudy today.",
    ]

    embeddings = embedder.embed_texts(texts=texts)

    similarity_1_2 = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    similarity_1_3 = cosine_similarity(
            embeddings[0],
            embeddings[2],
        )

    print()
    print("Text 1:")
    print(texts[0])

    print()
    print("Text 2:")
    print(texts[1])

    print()
    print("Text 3:")
    print(texts[2])

    print()
    print(f"Similarity (Text 1 vs Text 2): {similarity_1_2:.6f}")
    print(f"Similarity (Text 1 vs Text 3): {similarity_1_3:.6f}")


if __name__=="__main__":
    main()

