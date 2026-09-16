from advanced_rag_project.documents.loader import load_text_file
from advanced_rag_project.documents.chunker import chunk_text
from advanced_rag_project.retrieval.retriever import Retriever
from advanced_rag_project.rag.context_builder import build_context
from advanced_rag_project.rag.prompt_builder import build_rag_prompt

def main():

    print("=" * 50)
    print("RAG Prompt Builder Test")
    print("="  * 50)

    document = load_text_file(
        "data/documents/hindu_philosophy.txt"
    )

    chunks = chunk_text(
        document,
        chunk_size=200,
        chunk_overlap=30,
    )

    retriever = Retriever(chunks=chunks)

    question = "What doen advaita vedanta teach?"

    results = retriever.search(
        query=question,
        top_k=2,
    )

    context = build_context(
        results=results
    )

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    print("\n" + "=" * 50)
    print("QUESTION")
    print("=" * 50)
    print(question)

    print("\n" + "=" * 50)
    print("RAG PROMPT")
    print("=" * 50)
    print(prompt)

if __name__=="__main__":
    main()

