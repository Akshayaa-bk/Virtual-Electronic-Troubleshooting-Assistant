# === Imports ===
import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient

# LangChain utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

# Custom inventory modules
from inventory_handler import (
    extract_part_names,
    get_part_inventory,
    is_replacement_query
)


# === PDF Processing ===

def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF documents.

    Returns:
        documents_data (list): Dicts containing raw text, hash, and name
        doc_hashes (list): Unique hash of each document
        doc_names (list): Names of uploaded PDFs
    """
    documents_data, doc_hashes, doc_names = [], [], []

    for pdf in pdf_docs:
        content = pdf.getvalue()
        doc_hash = hashlib.md5(content).hexdigest()
        pdf_reader = PdfReader(pdf)

        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        documents_data.append({
            "text": pdf_text,
            "doc_hash": doc_hash,
            "doc_name": pdf.name
        })

        doc_hashes.append(doc_hash)
        doc_names.append(pdf.name)

    return documents_data, doc_hashes, doc_names


def get_text_chunks(documents_data):
    """
    Split full document text into smaller, manageable chunks.

    Args:
        documents_data (list): List of dicts containing doc text and metadata

    Returns:
        all_chunks (list): Chunks of text
        all_metadatas (list): Metadata for each chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
        length_function=len
    )

    all_chunks, all_metadatas = [], []
    for doc in documents_data:
        chunks = [chunk.strip() for chunk in splitter.split_text(doc["text"]) if chunk.strip()]
        metadata = [{"doc_hash": doc["doc_hash"], "doc_name": doc["doc_name"], "chunk_index": i} for i in range(len(chunks))]

        all_chunks.extend(chunks)
        all_metadatas.extend(metadata)

    return all_chunks, all_metadatas


# === BM25 Encoding ===

def load_bm25_encoder(file_path="bm25_values.json"):
    """
    Load previously saved BM25 encoder from disk.

    Args:
        file_path (str): Path to saved encoder file

    Returns:
        encoder (BM25Encoder or None): Loaded encoder or None if failed
    """
    if os.path.exists(file_path):
        try:
            encoder = BM25Encoder().default()
            encoder.load(file_path)
            return encoder
        except Exception as e:
            st.error(f"Error loading BM25 encoder: {e}")
    return None


def encode_text(text_chunks, existing_encoder=None):
    """
    Fit or reuse a BM25 encoder for given text chunks.

    Args:
        text_chunks (list): List of text chunks
        existing_encoder (BM25Encoder or None): Reuse existing if provided

    Returns:
        encoder (BM25Encoder): Fitted encoder
    """
    encoder = existing_encoder or BM25Encoder().default()
    encoder.fit(text_chunks)
    encoder.dump("bm25_values.json")
    return encoder


# === Hybrid Retrieval ===

def hybrid_search_retriever(text_chunks, metadatas, index, is_new_content=True):
    """
    Initialize and return a hybrid Pinecone retriever (dense + sparse).

    Args:
        text_chunks (list): Text data to index
        metadatas (list): Metadata for each chunk
        index (PineconeIndex): Pinecone index instance
        is_new_content (bool): Whether to add new chunks

    Returns:
        retriever (PineconeHybridSearchRetriever): Configured retriever
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    bm25_encoder = encode_text(text_chunks, load_bm25_encoder() if not is_new_content else None)

    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
        top_k=5
    )

    if is_new_content and text_chunks:
        retriever.add_texts(text_chunks, metadatas=metadatas)
        st.info(f"Added {len(text_chunks)} chunks to Pinecone")

    return retriever


def get_relevant_documents(query, retriever):
    """
    Retrieve top documents relevant to the user query.

    Args:
        query (str): User's input question
        retriever: Hybrid retriever instance

    Returns:
        documents (list): Ranked relevant documents
    """
    return retriever.get_relevant_documents(query)


def extract_document_sources(documents):
    """
    Count frequency of source documents in results.

    Args:
        documents (list): Retrieved documents

    Returns:
        sources (list): Sorted (doc_name, frequency) tuples
    """
    sources = {}
    for doc in documents:
        name = doc.metadata.get("doc_name", "Unknown")
        sources[name] = sources.get(name, 0) + 1

    return sorted(sources.items(), key=lambda x: x[1], reverse=True)


# === Inference Client ===

def initialize_inference_client():
    """
    Initialize Hugging Face inference client using token from .env.

    Returns:
        client (InferenceClient or None): Initialized client
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        st.error("Missing Hugging Face API token. Add HF_TOKEN to your environment.")
        return None

    try:
        return InferenceClient(provider="hf-inference", api_key=hf_token)
    except Exception as e:
        st.error(f"Error initializing InferenceClient: {str(e)}")
        return None


# === Query Handling ===

def process_query(user_question, retriever, Session):
    """
    Main query handler: checks inventory or generates answer using context.

    Args:
        user_question (str): Input from user
        retriever: Hybrid retriever instance
        Session: Active SQLAlchemy session

    Returns:
        response (str): Answer or inventory info
    """
    # Inventory check path
    if is_replacement_query(user_question):
        part_names = extract_part_names(user_question, Session)
        if part_names:
            inventory_info = []
            for part in part_names:
                items = get_part_inventory(part, Session)
                for item in items:
                    inventory_info.append(f"• {item.spare_part_name}: {item.quantity_available} available ({item.product_name})")
            if inventory_info:
                return "\n\n**Inventory Check:**\n" + "\n".join(inventory_info)

    # Retrieve relevant context
    docs = get_relevant_documents(user_question, retriever)
    if not docs:
        return "No relevant documents found to answer this question."

    source_docs = extract_document_sources(docs)
    primary_source = source_docs[0][0] if source_docs else "Unknown"
    context = "\n\n".join([doc.page_content for doc in docs]).strip()

    if not context:
        return "This information is not available in the documentation."

    # Generate LLM response
    client = initialize_inference_client()
    if not client:
        return "Error: Could not initialize Hugging Face Inference Client."

    prompt = f"""<|system|>
You are a technical assistant helping engineers troubleshoot and repair electrical equipment using provided documentation only.

Instructions:
- Use **only** the information available in the context.
- Do **not** make assumptions or fabricate information.
- If the documentation does not contain relevant info, respond: "This information is not available in the documentation."

Your response should include:
1. **Possible Cause(s)** of the reported issue
2. **Recommended Solution(s)** based on the documentation
3. Any **Replacement Part(s)** required (if applicable)

</s>
<|user|>
Context:
{context}

Reported Technical Issue:
{user_question}
</s>
<|assistant|>"""

    try:
        response = client.text_generation(
            prompt=prompt,
            model="HuggingFaceH4/zephyr-7b-beta",
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15
        )

        if response:
            return f"{response.strip()}\n\n*Source: {primary_source}*"

        return f"This information is not available in the documentation.\n\n*Source: {primary_source}*"

    except Exception as e:
        st.warning(f"Text generation failed: {str(e)}")
        return f"Error generating response: {str(e)}"


# === Index Cleanup ===

def clear_pinecone_index(index):
    """
    Clear all existing records from Pinecone index.

    Args:
        index (PineconeIndex): Pinecone index object
    """
    index.delete(delete_all=True)
    st.success("Pinecone index cleared successfully!")
