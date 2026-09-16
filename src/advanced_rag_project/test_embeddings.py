# from advanced_rag_project.embeddings.embedder import Embedder

# def main():

#     print("=" * 50)
#     print("Embeddding Test")
#     print("=" * 50)

#     embedder = Embedder()

#     text = "Advaita vedanta teachers non-duality"

#     embedding = embedder.embed_text(text=text)

#     print(f"Text: {text}")
#     print(f"Embeddings Dimensions: {len(embedding)}")
#     print(f"First 10 values: {embedding[:10]}")

# if __name__=="__main__":
#     main()


from advanced_rag_project.embeddings.embedder import Embedder


def main():
    print("=" * 50)
    print("Embedding Test")
    print("=" * 50)

    embedder = Embedder()

    texts = [
        "Advaita Vedanta teaches non-duality.",
        "Advaita Vedanta describes reality as non-dual.",
        "The weather is cloudy today.",
    ]

    embeddings = embedder.embed_texts(texts)

    for text, embedding in zip(texts, embeddings):
        print("\nText:")
        print(text)

        print(f"Dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()
