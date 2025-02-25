import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import time
import os
from dotenv import load_dotenv
import nltk 
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
P_API_KEY = os.getenv("P_API_KEY")

def pinecone_index(api_key):
    pc = Pinecone(api_key=api_key)
    index_name = "rag-electronics"

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(cloud='aws',region='us-east-1')
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    return pc.Index(index_name)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def encode_text(text_chunks):
    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(text_chunks)
    bm25_encoder.dump("bm25_values.json")
    return bm25_encoder

def hybrid_search_retriever(text_chunks,index):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    bm25_encoder = encode_text(text_chunks)
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index = index, top_k = 5)
    retriever.add_texts(text_chunks)
    return retriever

def get_conversational_chain(retriever):
    llm = HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        top_p=0.9
    )

    prompt_template = """
    You are a helpful AI assistant that answers questions based on the provided context.

    Context:
    {context}

    Question: {question}

    Answer the question based only on the provided context. If the information isn't in the context, say "I don't have information about that in my records."

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create RetrievalQA pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def process_query(user_question, retriever, chat_history):
    chain = get_conversational_chain(retriever)
    
    # For now, we're not using the chat history in the query
    # This is a simpler approach that will work with RetrievalQA
    response = chain.run(user_question)
    return response

def main():
    st.set_page_config(page_title="PDF Chat with Hybrid Search", layout="wide")
    st.header("Virtual Electronic Troubleshooting assistant💡📄")

    # Initialize session state for chat history if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    # Initialize Pinecone index
    index = pinecone_index(P_API_KEY)

    with st.sidebar:
        st.title("📂 Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Reset chat history when new documents are processed
                    st.session_state.chat_history = []
                    
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    retriever = hybrid_search_retriever(text_chunks, index)
                    st.session_state.retriever = retriever
                    st.session_state.processing_done = True
                    st.success("Documents processed! Start chatting now.")
            else:
                st.error("Please upload at least one PDF file.")
                
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDFs..."):
        if not st.session_state.processing_done:
            st.error("Please upload and process PDF documents first.")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = process_query(prompt, st.session_state.retriever, st.session_state.chat_history)
                    st.write(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()