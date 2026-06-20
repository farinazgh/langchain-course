#!/usr/bin/env python3
"""
rag_generation_from_pinecone.py

A clean RAG generation pipeline using:
- Pinecone as the vector database
- OpenAI embeddings for query embedding
- LangChain PineconeVectorStore for retrieval
- ChatOpenAI for answer generation

This script assumes the Pinecone index has already been populated by the
indexing pipeline.

Install:
    pip install -r requirements.txt

Environment variables:
    OPENAI_API_KEY="..."
    PINECONE_API_KEY="..."

Run:
    python rag_generation_from_pinecone.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class GenerationConfig:
    # Must match the Pinecone index created by the indexing pipeline.
    index_name: str = "cwc-rag-index"
    namespace: str = "wikipedia-2023-cricket-world-cup"

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    query: str = "Who won the 2023 Cricket World Cup?"
    top_k: int = 4

    temperature: float = 0.0
    max_retries: int = 2

    # Keep retrieved chunks compact when building the prompt.
    max_context_chars_per_chunk: int = 1600


CONFIG = GenerationConfig()


SYSTEM_PROMPT = """You are a careful RAG assistant.

Use only the provided context to answer the user's question.

Rules:
- If the answer is present in the context, answer clearly and directly.
- If the answer is not present in the context, say: "I don't know based on the provided context."
- Do not invent facts.
- Keep the answer concise.
- Mention which retrieved chunk numbers supported the answer when useful.
"""


USER_PROMPT_TEMPLATE = """Question:
{question}

Retrieved context:
{context}

Answer:
"""


def require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set.\n"
            f"Set it first, for example:\n"
            f"  export {name}='your-key-here'\n"
            f"Or put it in a .env file."
        )
    return value


def create_embeddings(config: GenerationConfig) -> OpenAIEmbeddings:
    require_env_var("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=config.embedding_model)


def create_llm(config: GenerationConfig) -> ChatOpenAI:
    require_env_var("OPENAI_API_KEY")
    return ChatOpenAI(
        model=config.chat_model,
        temperature=config.temperature,
        max_retries=config.max_retries,
    )


def create_pinecone_vector_store(
    config: GenerationConfig,
    embeddings: OpenAIEmbeddings,
) -> PineconeVectorStore:
    api_key = require_env_var("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(config.index_name):
        raise RuntimeError(
            f"Pinecone index '{config.index_name}' does not exist.\n"
            "Run the indexing pipeline first, or change GenerationConfig.index_name."
        )

    index = pc.Index(config.index_name)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )


def retrieve_context(
    vector_store: PineconeVectorStore,
    config: GenerationConfig,
) -> List[Tuple[Document, float]]:
    return vector_store.similarity_search_with_score(
        query=config.query,
        k=config.top_k,
        namespace=config.namespace,
    )


def format_doc_for_prompt(
    doc: Document,
    score: float,
    display_index: int,
    max_chars: int,
) -> str:
    text = doc.page_content.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."

    metadata = doc.metadata or {}
    source = metadata.get("source_url") or metadata.get("source") or "unknown source"
    chunk_index = metadata.get("chunk_index", "unknown")

    return (
        f"[Retrieved chunk {display_index}]\n"
        f"Source: {source}\n"
        f"Original chunk index: {chunk_index}\n"
        f"Similarity score: {score:.4f}\n"
        f"Content:\n{text}"
    )


def build_context(
    retrieved_docs: Sequence[Tuple[Document, float]],
    config: GenerationConfig,
) -> str:
    if not retrieved_docs:
        return "No context was retrieved."

    formatted_chunks = [
        format_doc_for_prompt(
            doc=doc,
            score=score,
            display_index=i,
            max_chars=config.max_context_chars_per_chunk,
        )
        for i, (doc, score) in enumerate(retrieved_docs, start=1)
    ]

    return "\n\n---\n\n".join(formatted_chunks)


def generate_answer(
    llm: ChatOpenAI,
    question: str,
    context: str,
) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    return str(response.content)


def print_retrieval_debug(
    retrieved_docs: Sequence[Tuple[Document, float]],
    config: GenerationConfig,
) -> None:
    print("\n=== Retrieved Documents ===")
    print(f"Namespace: {config.namespace}")
    print(f"Top-k: {config.top_k}")

    if not retrieved_docs:
        print("No documents retrieved.")
        return

    for i, (doc, score) in enumerate(retrieved_docs, start=1):
        text_preview = doc.page_content.replace("\n", " ").strip()
        if len(text_preview) > 300:
            text_preview = text_preview[:300] + "..."

        print(f"\n--- Result {i} ---")
        print(f"Score: {score:.4f}")
        print(f"Metadata: {doc.metadata}")
        print(f"Preview: {text_preview}")


def run(config: GenerationConfig) -> None:
    print("\n=== RAG Generation Pipeline: Pinecone Retrieval + OpenAI Generation ===")
    print(f"Pinecone index:  {config.index_name}")
    print(f"Namespace:       {config.namespace}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Chat model:      {config.chat_model}")
    print(f"Question:        {config.query}")

    embeddings = create_embeddings(config)
    vector_store = create_pinecone_vector_store(config, embeddings)

    retrieved_docs = retrieve_context(vector_store, config)
    print_retrieval_debug(retrieved_docs, config)

    context = build_context(retrieved_docs, config)

    llm = create_llm(config)
    answer = generate_answer(
        llm=llm,
        question=config.query,
        context=context,
    )

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Done ===\n")


if __name__ == "__main__":
    run(CONFIG)
