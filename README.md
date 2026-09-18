# Advanced RAG Project

An advanced Retrieval-Augmented Generation (RAG) system built with Python, Ollama, Sentence Transformers, and ChromaDB.

This project is the second stage of my local RAG learning journey. It builds on the foundational [`local_rag_project`](https://github.com/skv000/local_rag_project) and progressively introduces more advanced retrieval, document ingestion, persistence, reranking, context engineering, and generation techniques.

The goal is to understand how a production-oriented RAG system is designed by implementing and integrating each component step by step.

---

## Project Overview

This project starts from a working RAG-from-scratch implementation and progressively introduces more advanced components.

### Learning progression

```text
local_rag_project
RAG From Scratch
        │
        │ stable baseline
        ▼
advanced_rag_project
Advanced RAG Engineering
        │
        ├── Persistent Vector Storage
        ├── Document Ingestion Pipeline
        ├── Incremental Ingestion
        ├── Duplicate Detection
        ├── Modified Document Detection
        ├── Deleted Document Detection
        ├── File-Type Filtering
        ├── Recursive Directory Ingestion
        ├── Metadata Management
        ├── Dense Retrieval
        ├── Keyword Retrieval
        ├── Hybrid Retrieval
        ├── Cross-Encoder Reranking
        ├── Context Engineering
        ├── Grounded Generation
        ├── Source Attribution
        └── Chat History
                │
                ▼
        Next Engineering Stage
                │
                ├── RAG Evaluation
                ├── Query Transformation
                ├── Intelligent Source Routing
                ├── API Layer
                └── Observability & Deployment
````

---

# Current Architecture

The current system follows this pipeline:

```text
                    User Query
                        │
                        ▼
                ┌───────────────┐
                │ Hybrid Search │
                └───────┬───────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        Dense Retrieval     Keyword Retrieval
              │                   │
              └─────────┬─────────┘
                        │
                        ▼
                 Score Fusion
                        │
                        ▼
              Cross-Encoder Reranker
                        │
                        ▼
                Context Engineering
                        │
                        ▼
                 RAG Prompt Builder
                        │
                        ▼
                    Ollama
                        │
                        ▼
                   Final Answer
```

Documents are processed through:

```text
Documents
   │
   ▼
Document Loader
   │
   ▼
Text Normalization
   │
   ▼
Content Hashing
   │
   ▼
Stable Document ID
   │
   ▼
Chunking
   │
   ▼
Sentence Transformer Embeddings
   │
   ▼
ChromaDB
```

---

# Implemented Features

## 1. Persistent Vector Storage

The project uses ChromaDB for persistent vector storage.

Features include:

* Persistent collections
* Document-level identification
* Chunk-level storage
* Metadata storage
* Similarity search
* Document deletion
* Chunk reconstruction

Example storage structure:

```text
data/
└── documents/
    ├── hindu_philosophy.txt
    └── rag_test_document.txt

data/
└── chroma/
```

---

## 2. Document Ingestion

The ingestion pipeline supports:

* `.txt`
* `.md`

Documents are loaded, normalized, hashed, chunked, embedded, and stored in ChromaDB.

Example:

```python
from advanced_rag_project.ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline(
    persist_directory="data/chroma",
    collection_name="rag_documents",
)

stats = pipeline.ingest_directory("data/documents")

print(stats.to_dict())
```

---

## 3. Incremental Ingestion

The ingestion pipeline does not blindly re-ingest every document.

It detects:

* New documents
* Unchanged documents
* Modified documents
* Duplicate content
* Deleted documents
* Failed documents

Example ingestion statistics:

```text
Scanned:      2
New:          1
Unchanged:    0
Modified:     0
Deleted:      0
Duplicates:   1
Failed:       0
```

---

## 4. Content Hashing

Documents are assigned a SHA-256 content hash.

The hash is used to detect whether document content has changed and whether two different files contain identical content.

For example:

```text
hindu_philosophy.txt
Content Hash:
3fde8ff10ab51695d862d99d9a62bd850d6d76ceada2092a8da18afb94df5db3
```

This allows the ingestion pipeline to avoid storing duplicate content.

---

## 5. Stable Document IDs

Documents receive stable identifiers based on their source path.

Example:

```text
hindu_philosophy.txt
        ↓
hindu_philosophy
```

Document IDs allow the system to manage all chunks belonging to the same source document.

---

## 6. Duplicate Detection

If two documents have identical normalized content, only the first document is ingested.

Example:

```text
hindu_philosophy.txt
        │
        └── ingested

rag_test_document.txt
        │
        └── identical content
                ↓
             skipped
```

This prevents unnecessary duplicate vectors in the collection.

---

## 7. Modified Document Detection

If an already-ingested document changes, the pipeline detects the new content hash.

The old chunks are removed and the modified document is re-ingested.

```text
Existing document
       │
       ▼
Compare content hash
       │
       ├── Same → unchanged
       │
       └── Different
              │
              ▼
       Delete old chunks
              │
              ▼
       Re-ingest document
```

---

## 8. Deleted Document Detection

When ingesting a directory, the pipeline compares the documents currently present on disk with the documents stored in ChromaDB.

If an indexed document no longer exists, its chunks are removed.

---

## 9. Recursive Directory Ingestion

The ingestion pipeline recursively scans directories.

Example:

```text
data/documents/
├── philosophy.txt
├── notes.md
└── research/
    ├── paper1.txt
    └── paper2.md
```

All supported files can be discovered automatically.

Unsupported file types are ignored.

---

## 10. File-Type Filtering

Currently supported:

```text
.txt
.md
```

Unsupported file types are ignored by the ingestion pipeline.

The file-type system is intentionally simple so additional document loaders can be introduced later.

---

# Retrieval System

## 11. Dense Retrieval

Dense retrieval uses Sentence Transformers to generate embeddings.

Current embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

The resulting embeddings are stored in ChromaDB and used for semantic similarity search.

---

## 12. Keyword Retrieval

A lightweight keyword retrieval system was implemented using:

* Tokenization
* Term frequency
* Document frequency
* IDF
* TF-IDF-style scoring

This provides lexical matching alongside semantic retrieval.

---

## 13. Hybrid Retrieval

Dense and keyword retrieval are combined into a hybrid retrieval system.

Current default weighting:

```text
Dense retrieval   : 70%
Keyword retrieval : 30%
```

The system normalizes the individual retrieval scores before combining them.

Conceptually:

```text
Hybrid Score =
    Dense Weight × Dense Score
  + Keyword Weight × Keyword Score
```

This allows the system to benefit from both:

* Semantic similarity
* Exact keyword matching

---

## 14. Cross-Encoder Reranking

The hybrid retrieval candidates are passed to a cross-encoder reranker.

Current model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Pipeline:

```text
User Query
    │
    ▼
Hybrid Retrieval
    │
    ▼
Candidate Chunks
    │
    ▼
Cross Encoder
    │
    ▼
Reranked Chunks
```

The reranker evaluates the query and candidate document together rather than relying only on embedding similarity.

---

# Context Engineering

## 15. Context Engineering

The retrieved results are processed before being passed to the LLM.

The context engineering layer handles:

* Relevance ordering
* Duplicate removal
* Minimum relevance filtering
* Context token budgeting
* Source information
* Chunk metadata
* Context diagnostics

Example:

```text
Retrieved candidates
        │
        ▼
Sort by relevance
        │
        ▼
Remove duplicates
        │
        ▼
Apply relevance threshold
        │
        ▼
Apply context budget
        │
        ▼
Final LLM context
```

The system also reports diagnostics such as:

```text
Context items
Estimated tokens
Dropped duplicates
Dropped low-relevance chunks
Dropped budget-exceeded chunks
```

---

# Grounded Generation

## 16. RAG Prompt Engineering

The system builds a dedicated RAG prompt containing:

```text
CONTEXT
   ↓
QUESTION
   ↓
GROUNDING RULES
   ↓
ANSWER
```

The prompt instructs the model to use the supplied context when answering document-based questions.

---

## 17. Generation Result

Generation is represented using structured results containing information such as:

* Answer
* Grounded status
* Insufficient-context status
* Sources
* Context item count
* Estimated context tokens
* Unsupported sentences

This provides diagnostics beyond simply returning an LLM string.

---

## 18. Source Attribution

Answers generated from retrieved documents can expose their source documents.

Example:

```text
Sources:
  - data\documents\hindu_philosophy.txt
```

This makes it possible to trace an answer back to the documents used by the RAG pipeline.

---

# Chat History

The application maintains conversation history for conversational interactions.

For example:

```text
User:
What does Advaita Vedanta teach?

Assistant:
...

User:
What was my first question?

Assistant:
What does Advaita Vedanta teach?
```

Chat history is currently handled separately from document retrieval.

Future work will make the source-selection behavior more explicit between:

```text
Documents
Chat History
LLM Knowledge
```

---

# Current Knowledge Sources

The intended assistant architecture is:

```text
                  User Query
                      │
                      ▼
              Source Selection
                      │
          ┌───────────┼───────────┐
          │           │           │
          ▼           ▼           ▼
      Documents   Chat History   LLM
          │           │           │
          └───────────┼───────────┘
                      │
                      ▼
                   Answer
```

The desired behavior is:

### Document-based questions

Use information from uploaded documents.

```text
"What does my uploaded document say about Samkhya?"
```

### Chat-history questions

Use previous conversation context.

```text
"What was my first question?"
```

### General knowledge questions

When the relevant information is not present in the uploaded documents or conversation context, the system should be able to use the LLM's existing knowledge.

```text
"What is the capital of France?"
```

The explicit source-selection and fallback behavior is part of the next development stage.

---

# Technology Stack

| Component            | Technology                 |
| -------------------- | -------------------------- |
| Language             | Python 3.12+               |
| Package Manager      | uv                         |
| LLM Runtime          | Ollama                     |
| Local LLM            | Llama 3.2 3B               |
| Embeddings           | Sentence Transformers      |
| Embedding Model      | all-MiniLM-L6-v2           |
| Vector Store         | ChromaDB                   |
| Reranker             | Cross-Encoder              |
| Reranker Model       | ms-marco-MiniLM-L-6-v2     |
| Numerical Processing | NumPy                      |
| Hardware             | NVIDIA RTX 3050 Laptop GPU |

---

# Project Structure

```text
advanced_rag_project/
│
├── data/
│   ├── documents/
│   │   ├── hindu_philosophy.txt
│   │   └── rag_test_document.txt
│   │
│   └── chroma/
│
├── src/
│   └── advanced_rag_project/
│       │
│       ├── documents/
│       │   ├── chunker.py
│       │   ├── file_types.py
│       │   ├── hashing.py
│       │   ├── identifiers.py
│       │   ├── loader.py
│       │   └── models.py
│       │
│       ├── embeddings/
│       │   └── embedder.py
│       │
│       ├── ingestion/
│       │   ├── pipeline.py
│       │   └── results.py
│       │
│       ├── llm/
│       │   └── ollama_client.py
│       │
│       ├── rag/
│       │   ├── context_builder.py
│       │   ├── context_engineering.py
│       │   ├── generation.py
│       │   └── prompt_builder.py
│       │
│       ├── retrieval/
│       │   ├── chroma_retriever.py
│       │   ├── hybrid_retriever.py
│       │   ├── keyword_retriever.py
│       │   ├── models.py
│       │   ├── reranker.py
│       │   └── reranking_retriever.py
│       │
│       ├── routing/
│       │   └── router.py
│       │
│       ├── vectorstore/
│       │   └── chroma_store.py
│       │
│       └── main.py
│
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

---

# Running the Project

## 1. Install dependencies

```bash
uv sync
```

Make sure Ollama is installed and running.

The project currently uses:

```text
llama3.2:3b
```

---

## 2. Add documents

Place `.txt` or `.md` files inside:

```text
data/documents/
```

---

## 3. Ingest documents

```bash
uv run python -c "from advanced_rag_project.ingestion.pipeline import IngestionPipeline; p=IngestionPipeline(persist_directory='data/chroma', collection_name='rag_documents'); s=p.ingest_directory('data/documents'); print(s.to_dict())"
```

---

## 4. Run the application

```bash
uv run python -m advanced_rag_project.main
```

Example:

```text
YOU : What does Advaita Vedanta teach?

AI :
According to the provided context, Advaita Vedanta teaches that
ultimate reality is Brahman and that the individual self (Atman)
is ultimately not separate from Brahman.
```

---

# Testing

The project includes tests for individual components and integration stages.

Example:

```bash
uv run python src/advanced_rag_project/test_phase11_generation.py
```

Tests currently cover areas including:

* Chunking
* Embeddings
* Similarity
* Retrieval
* Hybrid retrieval
* Reranking
* Ingestion
* Context engineering
* Grounded generation

---

# Evaluation Roadmap

The next major milestone is **RAG evaluation**.

The evaluation system will measure the pipeline rather than relying only on manual inspection.

Planned evaluation areas:

```text
                    Evaluation
                        │
          ┌─────────────┼─────────────┐
          │             │             │
          ▼             ▼             ▼
      Retrieval      Context       Generation
          │             │             │
          ├── Hit@K     ├── Recall   ├── Correctness
          ├── Recall@K  ├── Precision├── Relevance
          ├── Precision ├── Duplicates├── Faithfulness
          └── MRR       └── Budget   └── Hallucination
```

The evaluation dataset will contain questions covering:

* Document-based questions
* Questions requiring specific chunks
* General LLM knowledge
* Chat-history questions
* Multi-turn questions
* Unsupported or ambiguous questions
* Questions combining documents and general knowledge

---

# Future Roadmap

## Retrieval

* Query transformation
* Query rewriting
* Multi-query retrieval
* Metadata-aware retrieval
* Advanced filtering
* Retrieval evaluation

## Generation

* Improved source-aware generation
* Better fallback behavior
* Answer evaluation
* Faithfulness evaluation
* Citation improvements

## Knowledge Routing

Explicitly determine whether a query should use:

```text
Document Knowledge
        │
        ├── Chat History
        │
        └── LLM Knowledge
```

## Production Engineering

* Evaluation framework
* Observability
* Logging
* Tracing
* API layer
* Authentication
* Document upload API
* Background ingestion
* Deployment
* Performance optimization

---

# Learning Philosophy

This project intentionally avoids immediately relying on high-level RAG frameworks.

The goal is to understand the individual components first:

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Search
   ↓
Keyword Search
   ↓
Hybrid Search
   ↓
Reranking
   ↓
Context Engineering
   ↓
Prompt Engineering
   ↓
LLM Generation
   ↓
Evaluation
```

Once these fundamentals are understood, higher-level frameworks and production infrastructure can be introduced with a clear understanding of what they are abstracting.

---

# Related Project

Foundational implementation:

[`local_rag_project`](https://github.com/skv000/local_rag_project)

This project focuses on progressively building a more advanced and production-oriented RAG architecture on top of that foundation.

---

# Project Status

**Current stage: Advanced RAG pipeline implemented — evaluation next.**

Implemented:

* Persistent document storage
* Incremental ingestion
* Duplicate detection
* Document modification detection
* Deleted document detection
* Recursive ingestion
* Dense retrieval
* Keyword retrieval
* Hybrid retrieval
* Cross-encoder reranking
* Context engineering
* Grounded generation
* Source attribution
* Conversation history

Next:

* RAG evaluation framework
* Source-aware routing
* LLM knowledge fallback
* Query transformation
* Production API
* Observability
* Deployment

````

