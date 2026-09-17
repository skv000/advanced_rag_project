from advanced_rag_project.documents.identifiers import create_document_id


def main():
    test_cases = {
        "data/documents/hindu_philosophy.txt": "hindu_philosophy",
        "data/documents/machine learning.txt": "machine_learning",
        "data/documents/RAG_Test.txt": "rag_test",
    }

    for file_path, expected in test_cases.items():

        document_id = create_document_id(file_path)

        print(
            f"{file_path} -> {document_id}"
        )

        assert document_id == expected

    print("\nAll document ID tests passed.")


if __name__ == "__main__":
    main()
    