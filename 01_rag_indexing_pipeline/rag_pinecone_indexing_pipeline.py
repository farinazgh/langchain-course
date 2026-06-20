#!/usr/bin/env python3
"""
rag_pinecone_indexing_pipeline.py

A clean, single-file RAG indexing pipeline using:
- LangChain WebBaseLoader to load a web page
- BeautifulSoupTransformer to clean the main article HTML
- RecursiveCharacterTextSplitter to create chunks
- OpenAIEmbeddings to create vectors
- Pinecone as the vector database

"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class RagConfig:
    url: str = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

    index_name: str = "cwc-rag-index"
    namespace: str = "wikipedia-2023-cricket-world-cup"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"

    embedding_model: str = "text-embedding-3-small"

    chunk_size: int = 1000
    chunk_overlap: int = 120
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    preview_count: int = 2
    preview_chars: int = 240

    clear_namespace_before_indexing: bool = False

    demo_query: str = "Who won the 2023 Cricket World Cup?"
    demo_k: int = 4


CONFIG = RagConfig()


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


def preview(label: str, items: List, count: int, chars: int) -> None:
    print(f"\n[{label}] Preview ({min(count, len(items))} of {len(items)})")
    for i in range(min(count, len(items))):
        text = items[i].page_content.replace("\n", " ").strip()
        suffix = "..." if len(text) > chars else ""
        print(f"  - {label.lower()}[{i}]: {text[:chars]}{suffix}")


def stable_chunk_ids(chunks: Iterable, *, url: str, namespace: str) -> List[str]:
    ids: List[str] = []
    for i, chunk in enumerate(chunks):
        raw = f"{url}|{namespace}|{i}|{chunk.page_content}".encode("utf-8")
        digest = hashlib.sha1(raw).hexdigest()[:16]
        ids.append(f"chunk-{i:05d}-{digest}")
    return ids


def get_index_dimension(description) -> int | None:
    if isinstance(description, dict):
        return description.get("dimension")
    return getattr(description, "dimension", None)


def get_index_ready(description) -> bool:
    status = (
        description.get("status", {})
        if isinstance(description, dict)
        else getattr(description, "status", {})
    )
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def create_embeddings(config: RagConfig) -> OpenAIEmbeddings:
    require_env_var("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=config.embedding_model)


def embedding_dimension(embeddings: OpenAIEmbeddings) -> int:
    return len(embeddings.embed_query("dimension check"))


def create_pinecone_client() -> Pinecone:
    api_key = require_env_var("PINECONE_API_KEY")
    return Pinecone(api_key=api_key)


def get_or_create_index(config: RagConfig, dimension: int):
    pc = create_pinecone_client()

    if not pc.has_index(config.index_name):
        print(f"[PINECONE] Creating index: {config.index_name}")
        pc.create_index(
            name=config.index_name,
            dimension=dimension,
            metric=config.pinecone_metric,
            spec=ServerlessSpec(
                cloud=config.pinecone_cloud,
                region=config.pinecone_region,
            ),
        )
    else:
        description = pc.describe_index(config.index_name)
        existing_dimension = get_index_dimension(description)
        if existing_dimension and existing_dimension != dimension:
            raise RuntimeError(
                f"Pinecone index '{config.index_name}' already exists with dimension "
                f"{existing_dimension}, but embedding model '{config.embedding_model}' "
                f"produces dimension {dimension}.\n\n"
                f"Fix: use a new index_name, or delete/recreate the existing index."
            )
        print(f"[PINECONE] Using existing index: {config.index_name}")

    for _ in range(30):
        description = pc.describe_index(config.index_name)
        if get_index_ready(description):
            break
        print("[PINECONE] Waiting for index to be ready...")
        time.sleep(2)

    return pc.Index(config.index_name)


def load_web_page(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No data loaded from URL: {url}")
    return docs


def clean_with_beautifulsoup(docs):
    transformer = BeautifulSoupTransformer()
    cleaned = transformer.transform_documents(
        docs,
        tags_to_extract=["div"],
        attrs_to_extract={"id": "mw-content-text"},
    )

    if cleaned and cleaned[0].page_content.strip():
        return cleaned
    return docs


def light_text_cleanup(docs):
    for doc in docs:
        doc.page_content = "\n".join(
            line.strip() for line in doc.page_content.splitlines() if line.strip()
        )
    return docs


def split_into_chunks(docs, config: RagConfig):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            **(chunk.metadata or {}),
            "source_url": config.url,
            "chunk_index": i,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }

    return chunks


def clear_namespace_if_requested(index, config: RagConfig) -> None:
    if not config.clear_namespace_before_indexing:
        return

    print(f"[PINECONE] Clearing namespace first: {config.namespace}")
    try:
        index.delete(delete_all=True, namespace=config.namespace)
    except Exception as exc:
        print(f"[PINECONE] Namespace clear warning: {exc}")


def build_pinecone_vector_store(index, embeddings: OpenAIEmbeddings) -> PineconeVectorStore:
    return PineconeVectorStore(index=index, embedding=embeddings)


def index_documents(
    vector_store: PineconeVectorStore,
    chunks,
    ids: List[str],
    config: RagConfig,
) -> None:
    print(f"[PINECONE] Upserting {len(chunks)} chunks into namespace: {config.namespace}")
    vector_store.add_documents(
        documents=chunks,
        ids=ids,
        namespace=config.namespace,
    )


def demo_similarity_search(vector_store: PineconeVectorStore, config: RagConfig) -> None:
    print(f"\n[QUERY] {config.demo_query}")
    results = vector_store.similarity_search(
        config.demo_query,
        k=config.demo_k,
        namespace=config.namespace,
    )

    print(f"[RESULTS] Returned: {len(results)} docs")
    for i, doc in enumerate(results, start=1):
        text = doc.page_content.replace("\n", " ").strip()
        print(f"\n--- Result {i} ---")
        print(text[:800] + ("..." if len(text) > 800 else ""))
        print(f"Metadata: {doc.metadata}")


def run(config: RagConfig) -> None:
    print("\n=== RAG Indexing Pipeline: Web → Clean → Chunk → OpenAI Embeddings → Pinecone ===")
    print(f"Source URL:      {config.url}")
    print(f"Pinecone index:  {config.index_name}")
    print(f"Namespace:       {config.namespace}")
    print(f"Embedding model: {config.embedding_model}")

    embeddings = create_embeddings(config)
    dimension = embedding_dimension(embeddings)
    print(f"[CONVERTER] Embeddings ready. Dimension: {dimension}")

    index = get_or_create_index(config, dimension)
    clear_namespace_if_requested(index, config)

    docs = load_web_page(config.url)
    print(f"\n[CONNECTOR] Loaded documents: {len(docs)}")
    print(f"[CONNECTOR] First doc chars raw: {len(docs[0].page_content)}")

    cleaned_docs = clean_with_beautifulsoup(docs)
    cleaned_docs = light_text_cleanup(cleaned_docs)
    print(f"[CLEAN] First doc chars cleaned: {len(cleaned_docs[0].page_content)}")
    preview("DOC", cleaned_docs, 1, config.preview_chars)

    chunks = split_into_chunks(cleaned_docs, config)
    if not chunks:
        raise RuntimeError("No chunks were created. Check the source content and splitter settings.")

    print(f"\n[SPLITTER] Chunks created: {len(chunks)}")
    preview("CHUNK", chunks, config.preview_count, config.preview_chars)

    ids = stable_chunk_ids(chunks, url=config.url, namespace=config.namespace)
    vector_store = build_pinecone_vector_store(index, embeddings)
    index_documents(vector_store, chunks, ids, config)

    stats = index.describe_index_stats()
    print(f"\n[PINECONE] Index stats: {stats}")

    demo_similarity_search(vector_store, config)
    print("\n=== Done ===\n")


if __name__ == "__main__":
    run(CONFIG)