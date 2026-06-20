# Simple RAG Pipeline

This project is a small example of a **Retrieval-Augmented Generation**, or **RAG**, system.

It has three main steps:

1. Index the data
2. Ask questions using RAG
3. Evaluate the RAG system

---

## What

This project shows how to build a basic RAG pipeline.

A RAG system connects an LLM to external knowledge. Instead of asking the model to answer only from its own memory, we first retrieve relevant text from a vector database and then give that text to the model as context.

The project has three files:

- `index.py` prepares and stores the data
- `rag.py` asks a question and generates an answer
- `evaluate.py` checks how good the RAG system is

---

## Why

LLMs can make mistakes or hallucinate when they do not have the right information.

RAG helps by giving the model relevant context before it answers.

This makes the answer:

- more grounded
- more factual
- easier to trace back to source data
- better suited for real documents, websites, or internal knowledge bases

Evaluation is also important because a RAG system should not only “work”; we need to measure whether it retrieves the right context and gives faithful answers.

---

## How

The system works in three stages.

First, the source data is loaded, split into smaller chunks, converted into embeddings, and stored in a vector database.

Then, when a user asks a question, the system searches the vector database for the most relevant chunks.

Finally, the retrieved chunks are passed to the LLM, and the LLM generates an answer based on that context.

The evaluation step tests the quality of this process.




## Summary

This project demonstrates the basic lifecycle of a RAG system:

```text
Prepare knowledge → Retrieve context → Generate answer → Evaluate quality
```