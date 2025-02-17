# LangChain Chroma Agent

This project demonstrates the use of LangChain with Chroma for document embedding and retrieval. It leverages Azure OpenAI for generating embeddings and executing chat-based interactions.

## Features

- **PDF Embedding**: Convert PDF documents into embeddings using Azure OpenAI
- **Chroma Vector Store**: Store and retrieve document embeddings using Chroma DB
- **Chunking**: Intelligent document chunking with customizable size and overlap
- **Chat Agent**: Interact with the system using a chat-based interface powered by LangChain and Azure OpenAI

## Prerequisites

- Python 3.8+
- Azure OpenAI account
- Chroma DB (running in Docker)

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Start Chroma DB**:
   ```bash
   docker run -d -p 8000:8000 -v C:/chroma/data:/vector_data -e CHROMA_SERVER_CORS_ALLOW_ORIGINS='["http://localhost:8090"]' -e PERSIST_DIRECTORY=/vector_data --name chromadb chromadb/chroma
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   - Copy `.env_sample` to `.env` and fill in the required API keys and endpoints.

   ### Environment Variables Description

   - `OPENAI_API_KEY`: Your OpenAI API key
   - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
   - `AZURE_OPENAI_ENDPOINT`: The endpoint URL for Azure OpenAI
   - `AZURE_OPENAI_API_VERSION`: The API version for Azure OpenAI
   - `LANGSMITH_TRACING`: Enable or disable LangSmith tracing (true/false)
   - `LANGSMITH_ENDPOINT`: The endpoint URL for LangSmith
   - `LANGSMITH_API_KEY`: Your LangSmith API key
   - `LANGSMITH_PROJECT`: The project name for LangSmith

5. **Prepare PDF Embeddings**:
   - Place your PDF files in the `./data/input/` directory
   - Run the embedding process to create and store document embeddings in Chroma DB

## Usage

1. **Start the Application**:
   ```bash
   python agent.py
   ```

2. **Interact with Documents**:
   - The system will process PDFs from the input directory
   - Use the chat interface to ask questions about your documents
   - The agent will retrieve relevant information using the Chroma vector store

## Configuration

You can customize the document processing by adjusting these parameters:
- Chunk size (default: 1000 characters)
- Chunk overlap (default: 200 characters)
- Collection name for vector storage

## Project Structure

- `agent.py`: Main script to run the chat agent
- `embeddings_tools.py`: Functions to create and manage document embeddings with Chroma
- `tools.py`: Utility functions for file writing and message extraction
- `.env_sample`: Sample environment configuration file
- `data/`: Directory for input PDFs and output markdown files

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the foundational framework
- [Chroma](https://www.trychroma.com/) for vector storage and retrieval
- [Azure OpenAI](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/) for embedding and chat capabilities