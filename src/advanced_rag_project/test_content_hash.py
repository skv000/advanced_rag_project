from advanced_rag_project.documents.hashing import calculate_content_hash


def main():
    text_a = "Hello world"
    text_b = "Hello world"
    text_c = "Hello World"

    hash_a = calculate_content_hash(text_a)
    hash_b = calculate_content_hash(text_b)
    hash_c = calculate_content_hash(text_c)

    print("Hash A:", hash_a)
    print("Hash B:", hash_b)
    print("Hash C:", hash_c)

    # Same content -> same hash
    assert hash_a == hash_b

    # Different content -> different hash
    assert hash_a != hash_c

    # SHA-256 produces a 64-character hexadecimal string
    assert len(hash_a) == 64

    print("\nAll content hash tests passed.")


if __name__ == "__main__":
    main()