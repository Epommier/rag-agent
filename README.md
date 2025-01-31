# LangChain Qdrant Agent

This project demonstrates the use of LangChain with Qdrant for document embedding and retrieval. It leverages Azure OpenAI for generating embeddings and executing chat-based interactions.

## Features

- **PDF Embedding**: Convert PDF documents into embeddings using Azure OpenAI.
- **Qdrant Vector Store**: Store and retrieve document embeddings using Qdrant.
- **Chat Agent**: Interact with the system using a chat-based interface powered by LangChain and Azure OpenAI.

## Prerequisites

- Python 3.8+
- Azure OpenAI account
- Qdrant account

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   - Copy `.env_sample` to `.env` and fill in the required API keys and endpoints.

   ### Environment Variables Description

   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key.
   - `AZURE_OPENAI_ENDPOINT`: The endpoint URL for Azure OpenAI.
   - `AZURE_OPENAI_API_VERSION`: The API version for Azure OpenAI.
   - `LANGSMITH_TRACING`: Enable or disable LangSmith tracing (true/false).
   - `LANGSMITH_ENDPOINT`: The endpoint URL for LangSmith.
   - `LANGSMITH_API_KEY`: Your LangSmith API key.
   - `LANGSMITH_PROJECT`: The project name for LangSmith.
   - `QDRANT_API_KEY`: Your Qdrant API key.
   - `QDRANT_URL`: The URL for your Qdrant instance.

4. **Prepare PDF Embeddings**:
   - Place your PDF files in the `./data/input/` directory.
   - Uncomment the lines in `agent.py` to create and upload PDF embeddings to Qdrant.

## Usage

1. **Run the Agent**:
   ```bash
   python agent.py
   ```

2. **Interact with the Agent**:
   - The agent will process the initial message and generate responses based on the embedded documents.

3. **Output**:
   - The responses are saved in the `data/output/` directory as markdown files.

## Project Structure

- `agent.py`: Main script to run the chat agent.
- `embeddings_tools.py`: Functions to create and upload PDF embeddings.
- `tools.py`: Utility functions for file writing and message extraction.
- `.env_sample`: Sample environment configuration file.
- `data/`: Directory for input PDFs and output markdown files.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the foundational framework.
- [Qdrant](https://qdrant.tech/) for vector storage and retrieval.
- [Azure OpenAI](https://azure.microsoft.com/en-us/services/cognitive-services/openai-service/) for embedding and chat capabilities.