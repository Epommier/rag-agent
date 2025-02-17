import os
import chromadb
import langchain

from uuid import uuid4
from typing import List, Dict, Any
from rich import print

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.globals import set_verbose
from langchain_community.document_loaders import PyMuPDFLoader

def create_pdf_chunks(
        pdf_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
    """
    Create chunks for a PDF file using LangChain text splitter.
    
    Args:
        pdf_path (str): Path to the PDF file        
        chunk_size (int): Size of text chunks (default: 1000)
        chunk_overlap (int): Overlap between chunks (default: 200)
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing text chunks and their metadata
    """
    # Load PDF
    loader = PyMuPDFLoader(
        pdf_path,
        extract_tables="markdown",
        extract_images=False)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    pages = []
    result = []

    for page in loader.lazy_load():        
        if len(pages) >= 100:
            break
        if page.page_content:
            pages.append(page)
            print(f":runner:[italic green] Processed page {page.metadata.get("page")}/{page.metadata.get("total_pages")}[/italic green]")
        else:
            print(f":warning:[italic yellow] No content on page {page.metadata.get("page")}/{page.metadata.get("total_pages")}[/italic yellow]")

    chunks = text_splitter.split_documents(pages)   

    # Create structure for each chunk
    for chunk in chunks:
        result.append({
            "text": chunk.page_content,
            "metadata": {
                "page": chunk.metadata.get("page"),
                "source": pdf_path if not chunk.metadata.get("title") else chunk.metadata["title"],
            }
        })
    
    return result

def upload_chunks(
    pdf_chunks: List[Dict[str, Any]],    
    embeddings: Embeddings,
    collection_name: str):
    """
    Uploads the given PDF chunks to a Vector collection.
    
    Args:
        pdf_chunks (List[Dict[str, Any]]): List of dictionaries containing text chunks and their embeddings.
        embeddings (Embeddings): Embeddings to use
        collection_name (str): Name of the collection where the chunks will be uploaded.
    
    Returns:
        None
    """

    vector_store = get_vector_store(collection_name, embeddings, True)
    docs = [Document(page_content=chunk['text'], metadata=chunk['metadata'], id=str(uuid4())) for chunk in pdf_chunks] 
    ids = vector_store.add_documents(docs, ids=[doc.id for doc in docs])

    return ids

def get_vector_store(
    collection_name: str,
    embeddings: Embeddings,
    recreate_col: bool = False) -> Chroma:
    """
    Get a Chroma vector store from the given collection name and embeddings.
    
    If recreate_col is set to True, the collection will be deleted and recreated.
    
    Args:
        collection_name (str): Name of the collection
        embeddings (Embeddings): Embeddings to use
        recreate_col (bool, optional): Whether to recreate the collection. Defaults to False.
    
    Returns:
        Chroma: Chroma vector store
    """    
    chroma_client = chromadb.HttpClient(
        host=os.environ.get("CHROMADB_HOST"),
        port=os.environ.get("CHROMADB_PORT")
    )

    if recreate_col:
        col = chroma_client.get_collection(name=collection_name)
        if col:
            print(f"[bold green]Found {len(col.get()["documents"])} chunks in the collection[/]")
            chroma_client.delete_collection(name=collection_name)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=chroma_client
    )

    return vector_store

if __name__ == "__main__":
    set_verbose(True)

    print(f"Langchain version: {langchain.__version__}")
    print(f"Chroma version: {chromadb.__version__}")

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"]
    )

    input_folder = "./data/input/"
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_chunks = create_pdf_chunks(os.path.join(input_folder, filename))
            upload_chunks(pdf_chunks, embeddings, os.environ["VECTOR_COLLECTION"])