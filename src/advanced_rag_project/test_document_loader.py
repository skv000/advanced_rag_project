from advanced_rag_project.documents.loader import load_text_file

def main():
    document = load_text_file(
        "data/documents/hindu_philosophy.txt"
    )

    print("="*50)
    print("Document Loader test")
    print("="*50)

    print(document)

if __name__=="__main__":
    main()

