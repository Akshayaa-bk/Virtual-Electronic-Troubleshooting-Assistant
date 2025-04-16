import time
from pinecone import Pinecone, ServerlessSpec

def pinecone_index(api_key):
    """
    Initialize and return a Pinecone index for vector storage and retrieval.

    This function checks if an index named 'electronics' exists. If not,
    it creates a new serverless index with a vector dimension of 384
    (suitable for MiniLM or similar embedding models) and waits until the
    index is fully initialized before returning a reference to it.

    Args:
        api_key (str): Pinecone API key.

    Returns:
        pinecone.Index: An active Pinecone index instance.
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)

    # Set index name
    index_name = "electronics"

    # Check if index exists; create if it doesn't
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Embedding vector size
            metric="dotproduct",  # Similarity metric
            spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Deployment config
        )

        # Wait for the index to become ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # Return a reference to the index
    return pc.Index(index_name)


def fetch_stored_documents(index):
    """
    Retrieve all unique document names stored as metadata in the index.

    This function queries the index with a dummy zero vector to get a wide
    sweep of available vectors and extract document names from their metadata.

    Args:
        index (pinecone.Index): The Pinecone index to query.

    Returns:
        list[str]: A list of unique document names stored in the index.
    """
    stored_documents = set()

    # Query with zero-vector just to retrieve metadata of all vectors
    response = index.query(
        vector=[0] * 384,
        top_k=10000,  # Adjust based on your index size
        include_metadata=True
    )

    # Extract unique doc_name values from metadata
    for match in response.get('matches', []):
        doc_name = match.get("metadata", {}).get("doc_name")
        if doc_name:
            stored_documents.add(doc_name)

    return list(stored_documents)
