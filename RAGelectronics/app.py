import streamlit as st
import time
import os
import pandas as pd
import nltk
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sqlalchemy import text

# Import custom modules for document processing
from document_processing import (
    get_pdf_text, get_text_chunks, 
    hybrid_search_retriever, get_relevant_documents,
    extract_document_sources,
    process_query, clear_pinecone_index, load_bm25_encoder
)

# Import custom modules for inventory management
from inventory_handler import (
    setup_database, import_inventory_from_excel, search_inventory,
    get_part_inventory, update_inventory_quantity,
    get_inventory_stats, extract_part_names, is_replacement_query
)

# Import Pinecone utilities
from pinecone_utils import pinecone_index, fetch_stored_documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder


# ----- INITIALIZATION FUNCTIONS -----

def initialize_app():
    """
    Initialize the application's environment and dependencies.
    """
    # Load environment variables
    load_dotenv()
    
    # Setup API keys from environment
    hf_token = os.getenv("HF_TOKEN")
    p_api_key = os.getenv("P_API_KEY")
    
    # Download required NLTK resources
    nltk.download('punkt')
    nltk.download('wordnet')
    
    return hf_token, p_api_key


def initialize_session_state():
    """
    Initialize all Streamlit session state variables.
    """
    # Chat and document retrieval states
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    
    # Inventory management states
    if "inventory_updated" not in st.session_state:
        st.session_state.inventory_updated = False
    
    if "inventory_action" not in st.session_state:
        st.session_state.inventory_action = None
    
    if "replace_inventory" not in st.session_state:
        st.session_state.replace_inventory = False
    
    if "clear_inventory_flag" not in st.session_state:
        st.session_state.clear_inventory_flag = False


# ----- UTILITY FUNCTIONS -----

def clear_chat_history():
    """
    Clear the chat history from session state.
    """
    st.session_state.chat_history = []
    st.success("Chat history cleared!")


def initialize_retriever(index):
    """
    Initialize document retriever from existing Pinecone index.
    
    Args:
        index: Pinecone index object
        
    Returns:
        bool: True if retriever was initialized successfully
    """
    try:
        # Set up embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load or create BM25 encoder for sparse embeddings
        bm25_encoder = load_bm25_encoder()
        if not bm25_encoder:
            bm25_encoder = BM25Encoder().default()
            bm25_encoder.dump("bm25_values.json")
        
        # Create hybrid retriever combining dense and sparse embeddings
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index,
            top_k=5
        )
        
        # Store in session state
        st.session_state.retriever = retriever
        st.session_state.processing_done = True
        return True
        
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        return False


# ----- DOCUMENT PROCESSING CALLBACKS -----

def process_docs_callback(pdf_docs, index, stored_docs):
    """
    Process uploaded PDF documents and add them to the vector index.
    
    Args:
        pdf_docs: List of uploaded PDF files
        index: Pinecone index object
        stored_docs: List of already stored document names
    """
    if pdf_docs:
        with st.spinner("Processing Technical Documentation..."):
            # Extract text from PDFs
            documents_data, doc_hashes, doc_names = get_pdf_text(pdf_docs)
            
            # Filter out documents that are already in the index
            stored_docs_set = set(stored_docs)
            new_docs = [doc for doc in documents_data if doc["doc_name"] not in stored_docs_set]

            if new_docs:
                # Process and index new documents
                text_chunks, metadatas = get_text_chunks(new_docs)
                retriever = hybrid_search_retriever(text_chunks, metadatas, index, is_new_content=True)
                st.session_state.retriever = retriever
                st.session_state.processing_done = True
                st.success(f"Processed {len(new_docs)} new manual(s)!")
            else:
                st.success("No new documents detected.")


def reset_database_callback(index):
    """
    Clear the document database from Pinecone.
    
    Args:
        index: Pinecone index object
    """
    with st.spinner("Clearing document database..."):
        clear_pinecone_index(index)
        st.session_state.retriever = None
        st.session_state.processing_done = False
        st.success("Technical database reset complete!")


# ----- INVENTORY MANAGEMENT FUNCTIONS -----

def display_inventory_dataframe(items, title=None):
    """
    Display inventory items in a DataFrame.
    
    Args:
        items: List of InventoryItem objects
        title: Optional title to display above DataFrame
    
    Returns:
        DataFrame: The created pandas DataFrame
    """
    try:
        if title:
            st.subheader(title)
            
        # Convert items to DataFrame
        df = pd.DataFrame([
            {
                "ID": item.id,
                "Product": item.product_name,
                "Spare Part Name": item.spare_part_name,
                "Description": item.description,
                "Quantity Available": item.quantity_available,
                "Location": getattr(item, 'location', 'N/A'),
                "Part Number": getattr(item, 'part_number', 'N/A'),
                "Price": getattr(item, 'price', 'N/A')
            }
            for item in items
        ])
        
        # Display the DataFrame
        st.dataframe(df, use_container_width=True)
        return df
        
    except Exception as e:
        st.error(f"Error displaying inventory data: {str(e)}")
        return None


def create_download_button(df, filename, button_text):
    """
    Create a download button for a DataFrame.
    
    Args:
        df: Pandas DataFrame to download
        filename: Name of the output file
        button_text: Text to display on the button
    """
    try:
        # Convert DataFrame to CSV
        csv = df.to_csv(index=False).encode('utf-8')
        
        # Create download button
        st.download_button(
            button_text,
            csv,
            filename,
            "text/csv",
            key=f'download-{filename}'
        )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")


def import_inventory_callback(excel_file, engine, Session):
    """
    Import inventory data from Excel file.
    
    Args:
        excel_file: Uploaded Excel file
        engine: SQLAlchemy engine
        Session: SQLAlchemy session maker
    """
    if excel_file:
        with st.spinner("Importing inventory data..."):
            import_inventory_from_excel(excel_file, engine, Session, clear_inventory=False)
            st.session_state.inventory_updated = True
            st.session_state.inventory_action = "imported"
            st.rerun()
    else:
        st.error("Please upload an Excel file first.")


def update_quantity_callback(item_id, new_quantity, Session):
    """
    Update inventory quantity.
    
    Args:
        item_id: ID of the inventory item
        new_quantity: New quantity value
        Session: SQLAlchemy session maker
    """
    if update_inventory_quantity(item_id, new_quantity, Session):
        st.session_state.inventory_updated = True
        st.session_state.inventory_action = "updated"
        st.rerun()
    else:
        st.session_state.inventory_action = "failed"


def clear_inventory_callback(Session):
    """
    Clear all inventory data.
    
    Args:
        Session: SQLAlchemy session maker
    """
    session = Session()
    session.execute(text("DELETE FROM inventory"))
    session.commit()
    session.close()
    st.session_state.clear_inventory_flag = False
    st.session_state.inventory_updated = True
    st.session_state.inventory_action = "cleared"
    st.rerun()


# ----- UI COMPONENTS -----

def render_troubleshooting_tab(index, Session):
    """
    Render the troubleshooting chat tab.
    
    Args:
        index: Pinecone index object
        Session: SQLAlchemy session maker
    """
    # Sidebar for document management
    with st.sidebar:
        st.title("Technical Documentation")

        # Fetch and display indexed documents
        stored_docs = fetch_stored_documents(index)
        if stored_docs:
            st.session_state.processing_done = True  # Documents are already indexed

        st.subheader("Indexed Technical Manuals")
        for doc_name in stored_docs:
            st.text(f"• {doc_name}")

        # Document upload section
        pdf_docs = st.file_uploader("Upload Technical Manuals (PDF)", accept_multiple_files=True)

        # Document management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Process Documents"):
                process_docs_callback(pdf_docs, index, stored_docs)

        with col2:
            if st.button("Reset Database"):
                reset_database_callback(index)
                
        # Clear chat history button
        if st.button("Clear Chat History"):
            clear_chat_history()

    # Chat container
    chat_container = st.container()
    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input for chat
    if prompt := st.chat_input("Enter technical issue details..."):
        st.chat_message("user").write(prompt)
        
        if not st.session_state.processing_done and not st.session_state.retriever:
            st.error("Please upload and process technical documentation first.")
        else:
            with st.spinner("Analyzing issue..."):
                # Initialize retriever if needed
                if not st.session_state.retriever:
                    if stored_docs:
                        initialize_retriever(index)
                
                # Process user query and generate response
                response = process_query(prompt, st.session_state.retriever, Session)
                
                # Update chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Display assistant response
                st.chat_message("assistant").write(response)


def render_inventory_tab(engine, Session):
    """
    Render the inventory management tab.
    
    Args:
        engine: SQLAlchemy engine
        Session: SQLAlchemy session maker
    """
    st.header("Inventory Management")
    
    # Display toast message for inventory updates
    if st.session_state.inventory_updated:
        st.session_state.inventory_updated = False
        st.toast("Inventory updated!", icon="✅")

    # Inventory statistics section
    st.subheader("📊 Inventory Overview")
    stats = get_inventory_stats(Session)

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Items", stats["total_items"])
    
    with col2:
        st.metric("Low Stock Items", stats["low_stock"], 
                  delta=f"{stats['low_stock']} need attention" if stats["low_stock"] > 0 else None,
                  delta_color="inverse")
    
    with col3:
        st.metric("Out of Stock", stats["out_of_stock"],
                 delta=f"{stats['out_of_stock']} need reordering" if stats["out_of_stock"] > 0 else None,
                 delta_color="inverse")

    # Fetch inventory data for different categories
    low_stock_items = search_inventory("low_stock", Session)
    out_of_stock_items = search_inventory("out_of_stock", Session)

    # Create tabs for different inventory views
    inv_tab1, inv_tab2, inv_tab3 = st.tabs(["Low Stock Items", "Out of Stock Items", "All Inventory"])
    
    # Low stock items tab
    with inv_tab1:
        if low_stock_items:
            df_low_stock = display_inventory_dataframe(
                low_stock_items, 
                title="⚠️ Low Stock Items (5 or fewer)"
            )
            
            if df_low_stock is not None:
                create_download_button(
                    df_low_stock,
                    "low_stock_report.csv",
                    "Download Low Stock Report"
                )
        else:
            st.success("No low stock items - inventory levels are good!")

    # Out of stock items tab
    with inv_tab2:
        if out_of_stock_items:
            df_out_of_stock = display_inventory_dataframe(
                out_of_stock_items,
                title="❌ Out of Stock Items"
            )
            
            if df_out_of_stock is not None:
                create_download_button(
                    df_out_of_stock,
                    "out_of_stock_report.csv",
                    "Download Out of Stock Report"
                )
        else:
            st.success("No out of stock items!")
    
    # All inventory tab with search
    with inv_tab3:
        st.subheader("🔍 Search Inventory")
        search_term = st.text_input("Search for parts, products, or descriptions", key="search_inventory")
        
        # Display search results or all inventory
        if search_term:
            results = search_inventory(search_term, Session)
            if results:
                st.success(f"Found {len(results)} matching items")
            else:
                st.info("No matching items found.")
        else:
            results = search_inventory("", Session)  # Empty search returns all items
            
        # Display inventory data
        if results:
            df = display_inventory_dataframe(results)
            
            if df is not None:
                create_download_button(
                    df,
                    "inventory_report.csv",
                    "Download Inventory Report"
                )
    
    # Import/Export Section
    st.markdown("---")
    st.subheader("📥 Import Inventory")
    
    excel_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="excel_uploader")
    
    # Clear inventory with confirmation
    clear_inventory_col, import_col = st.columns(2)
    
    with clear_inventory_col:
        if st.button("🗑️ Clear Inventory"):
            st.session_state.clear_inventory_flag = True
            
        # Confirmation dialog for clearing inventory
        if st.session_state.clear_inventory_flag:
            st.warning("⚠️ Are you sure you want to clear ALL inventory data? This cannot be undone.")
            clear_col1, clear_col2 = st.columns(2)
            
            with clear_col1:
                if st.button("✅ Yes, Clear All Data"):
                    clear_inventory_callback(Session)
            
            with clear_col2:
                if st.button("❌ Cancel"):
                    st.session_state.clear_inventory_flag = False
                    st.rerun()
        
    # Import inventory button
    with import_col:
        if st.button("📥 Import Inventory", disabled=(excel_file is None)):
            import_inventory_callback(excel_file, engine, Session)
    
    # Display success messages for inventory actions
    if st.session_state.inventory_action == "imported":
        st.success("✅ Inventory successfully imported!")
        st.session_state.inventory_action = None
    elif st.session_state.inventory_action == "cleared":
        st.success("✅ Inventory database has been cleared!")
        st.session_state.inventory_action = None
        
    # Inventory Update Section
    st.markdown("---")
    st.subheader("🔄 Update Inventory Quantity")
    
    # Update form with three columns
    update_col1, update_col2, update_col3 = st.columns(3)
    
    with update_col1:
        item_id = st.number_input("Item ID", min_value=1, step=1, key="item_id_input")
    
    with update_col2:
        new_quantity = st.number_input("New Quantity", min_value=0, step=1, key="quantity_input")
    
    with update_col3:
        if st.button("🔄 Update Quantity", key="update_button"):
            update_quantity_callback(item_id, new_quantity, Session)
            
    # Show appropriate message based on the action result
    if st.session_state.inventory_action == "updated":
        st.success(f"✅ Updated quantity for item ID {item_id}")
        st.session_state.inventory_action = None
    elif st.session_state.inventory_action == "failed":
        st.error(f"❌ Failed to update item ID {item_id}. Item ID not found.")
        st.session_state.inventory_action = None


# ----- MAIN APPLICATION -----

def main():
    """
    Main application function.
    """
    # Configure page settings
    st.set_page_config(page_title="Technical Troubleshooting Platform", layout="wide")
    st.header("Electronics Engineering Troubleshooting Platform 🔧🔌")

    # Initialize environment and session state
    hf_token, p_api_key = initialize_app()
    initialize_session_state()

    # Setup database connection
    engine, Session = setup_database()

    # Initialize Pinecone index
    index = pinecone_index(p_api_key)

    # Create tabs for different functionality
    tab1, tab2 = st.tabs(["Troubleshooting Chat", "Inventory Management"])
    
    # Render troubleshooting tab
    with tab1:
        render_troubleshooting_tab(index, Session)
    
    # Render inventory management tab
    with tab2:
        render_inventory_tab(engine, Session)


if __name__ == "__main__":
    main()