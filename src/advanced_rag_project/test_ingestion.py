from advanced_rag_project.ingestion.pipeline import IngestionPipeline


def main():
    pipeline = IngestionPipeline(
        persist_directory="data/chroma_ingestion_test",
        collection_name="documents",
    )

    pipeline.ingest_file(
        "data/documents/rag_test_document.txt"
    )


if __name__ == "__main__":
    main()