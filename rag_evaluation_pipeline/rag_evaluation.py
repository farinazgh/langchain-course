# evaluate.py

import os
import time
from dotenv import load_dotenv

from datasets import Dataset

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    AnswerCorrectness,
    ResponseRelevancy,
    FactualCorrectness,
)

# --- CONFIG ---
URL = "https://en.wikipedia.org/wiki/2023_Cricket_World_Cup"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TEST_SIZE = 5
RETRIEVAL_K = 4


INDEX_NAME = "cwc-index"
NAMESPACE = "evaluation"

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

DELETE_EXISTING_NAMESPACE = True


# --- LOAD ENV ---
load_dotenv()


# --- HELPERS ---
def get_first_existing_value(row, possible_columns):
    for column in possible_columns:
        if column in row and row[column] is not None:
            value = row[column]

            if isinstance(value, float) and str(value) == "nan":
                continue

            return value

    raise KeyError(
        f"None of these columns were found in the generated testset: {possible_columns}"
    )


def normalize_reference(reference):
    if isinstance(reference, list):
        return " ".join(str(item) for item in reference)

    return str(reference)


def generate_answer(llm, question, contexts):
    context_text = "\n\n".join(contexts)

    messages = [
        (
            "system",
            "You are a factual RAG assistant. Answer the user's question using only the provided context. "
            "If the answer is not in the context, say that you do not know.",
        ),
        (
            "human",
            f"Question:\n{question}\n\nContext:\n{context_text}",
        ),
    ]

    response = llm.invoke(messages)
    return response.content


# --- PINECONE SETUP ---
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

print("Checking Pinecone index...")
if not pc.has_index(INDEX_NAME):
    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    print("Waiting for Pinecone index to be ready...")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)

else:
    print(f"Pinecone index already exists: {INDEX_NAME}")

index = pc.Index(INDEX_NAME)


# --- LOAD AND SPLIT DOCUMENTS ---
print("Loading document...")
documents = WebBaseLoader(URL).load()

print("Splitting document...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunks = splitter.split_documents(documents)


# --- MODELS ---
print("Creating LLM and embeddings...")
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

evaluator_llm = LangchainLLMWrapper(llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)


# --- VECTOR STORE ---
print("Connecting LangChain to Pinecone...")
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=NAMESPACE,
)

if DELETE_EXISTING_NAMESPACE:
    print(f"Cleaning Pinecone namespace: {NAMESPACE}")
    try:
        index.delete(delete_all=True, namespace=NAMESPACE)
        time.sleep(3)
    except Exception as e:
        print(f"Namespace cleanup skipped or failed: {e}")

print("Uploading chunks to Pinecone...")
vector_store.add_documents(chunks)

print("Waiting briefly for Pinecone indexing...")
time.sleep(5)


# --- GENERATE TESTSET ---
print("Generating testset...")
generator = TestsetGenerator(
    llm=evaluator_llm,
    embedding_model=evaluator_embeddings,
)

try:
    testset = generator.generate_with_langchain_docs(
        chunks,
        testset_size=TEST_SIZE,
    )
except TypeError:
    # Older RAGAS versions used test_size instead of testset_size
    testset = generator.generate_with_langchain_docs(
        chunks,
        test_size=TEST_SIZE,
    )

testset_df = testset.to_pandas()

print("\nGenerated testset columns:")
print(list(testset_df.columns))


# --- RUN REAL RAG PIPELINE ---
print("Running RAG pipeline over Pinecone...")

evaluation_rows = []

for _, row in testset_df.iterrows():
    question = get_first_existing_value(
        row,
        ["user_input", "question", "query"],
    )

    reference = get_first_existing_value(
        row,
        ["reference", "ground_truth", "ground_truths", "answer"],
    )

    reference = normalize_reference(reference)

    retrieved_docs = vector_store.similarity_search(
        question,
        k=RETRIEVAL_K,
    )

    retrieved_contexts = [doc.page_content for doc in retrieved_docs]

    answer = generate_answer(
        llm=llm,
        question=question,
        contexts=retrieved_contexts,
    )

    evaluation_rows.append(
        {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
        }
    )


dataset = Dataset.from_list(evaluation_rows)


# --- EVALUATE ---
print("Evaluating with RAGAS...")

result = evaluate(
    dataset=dataset,
    metrics=[
        LLMContextRecall(),
        Faithfulness(),
        AnswerCorrectness(),
        ResponseRelevancy(),
        FactualCorrectness(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)


print("\n=== RAGAS RESULT ===\n")
print(result)

print("\n=== RAGAS RESULT DATAFRAME ===\n")
print(result.to_pandas())
