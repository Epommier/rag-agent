import os

from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict, Any

def create_pdf_embeddings(pdf_path: str, embeddings: AzureOpenAIEmbeddings, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Create embeddings for a PDF file using LangChain text splitter and Azure OpenAI embeddings.
    
    Args:
        pdf_path (str): Path to the PDF file
        embeddings (AzureOpenAIEmbeddings): Azure OpenAI embeddings
        chunk_size (int): Size of text chunks (default: 1000)
        chunk_overlap (int): Overlap between chunks (default: 200)
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing text chunks and their embeddings
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(pages)
    
    # Create embeddings for each chunk
    result = []
    for chunk in chunks:
        vector = embeddings.embed_query(chunk.page_content)
        result.append({
            "text": chunk.page_content,
            "metadata": {
                "page": chunk.metadata.get("page"),
                "source": pdf_path,
            },
            "vector": vector,
        })
    
    return result

def upload_chunks_to_qdrant(pdf_chunks: List[Dict[str, Any]], collection_name: str):
    """
    Uploads the given PDF chunks to a Qdrant collection.
    
    Args:
        pdf_chunks (List[Dict[str, Any]]): List of dictionaries containing text chunks and their embeddings.
        collection_name (str): Name of the Qdrant collection where the chunks will be uploaded.
    
    Returns:
        None
    """
    # Initialize Qdrant client
    qdrant_client = QdrantClient(
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY") 
    )

    # Create the collection in Qdrant if it doesn't exist
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(pdf_chunks[0]['vector']), distance=Distance.COSINE)
    )

    # Prepare the data for insertion
    vectors = [chunk['vector'] for chunk in pdf_chunks]
    payloads = [{"text": chunk['text'], "metadata": chunk['metadata']} for chunk in pdf_chunks]

    # Insert the data into the Qdrant collection
    qdrant_client.upload_collection(
        collection_name=collection_name,
        vectors=vectors,
        payload=payloads,
        ids=None  # Let Qdrant auto-generate IDs
    )
