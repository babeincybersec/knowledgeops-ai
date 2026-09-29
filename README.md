# KnowledgeOps AI

A production-oriented RAG (Retrieval-Augmented Generation) knowledge assistant
that ingests company documents, answers questions with citations, evaluates its
own responses, and updates its knowledge base automatically.

> **Status:** 🚧 Milestone 0 — Engineering setup

---

## What It Does

KnowledgeOps AI solves a real problem: employees waste time searching through
scattered documentation, and naive LLM chatbots hallucinate information that
isn't in company documents.

This system:

- **Ingests** PDFs, text files, and HTML documents
- **Indexes** them into a vector database
- **Retrieves** relevant passages for each question
- **Generates** answers grounded in retrieved context
- **Cites** sources so answers are verifiable
- **Evaluates** its own performance with metrics
- **Updates** its knowledge base when documents change

---

## Architecture

```text
Documents → Ingestion → Chunks → Embeddings → Vector DB
                                                    ↓
User Question → Embedding → Retrieval → LLM → Answer + Citations
                                                    ↓
                                               Evaluation
```

## Evaluation

KnowledgeOps AI is evaluated against a hand-crafted 15-case benchmark
(11 answerable, 4 unanswerable) built from the sample employee handbook.

| Metric | Score |
|---|---|
| Retrieval hit rate | 11/11 (100%) |
| Answer accuracy | 8/11 (73%) |
| Faithfulness (LLM-as-judge) | 11/11 (100%) |
| Not-found accuracy | 4/4 (100%) |
| Citation presence | 11/11 (100%) |

Total score: 95%

Run the evaluation yourself:

```bash
python -m evaluation.evaluator