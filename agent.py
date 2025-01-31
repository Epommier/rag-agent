import os
import uuid

from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from langgraph.prebuilt import create_react_agent

from embeddings_tools import create_pdf_embeddings, upload_chunks_to_qdrant
from tools import write_to_file, extract_message_content
from typing import Annotated, Any, Dict, List
from pydantic import BaseModel, Field

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState

from langchain.vectorstores import Qdrant
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import AIMessage, ToolMessage, AIMessageChunk, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.globals import set_verbose
from langchain.prompts import MessagesPlaceholder
from langchain.tools import tool

set_verbose(True)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="text-embedding-3-large")

collection_name = "pdf_embeddings"
#pdf_chunks = create_pdf_embeddings("./data/input/PatternsMartinFowler.pdf", embeddings)
#upload_chunks_to_qdrant(pdf_chunks, collection_name)

class Reference(BaseModel):
    """A reference to a document source"""
    content: str = Field(..., description="The content of the reference")
    title: str = Field(..., description="The title of the reference")
    page: str = Field(..., description="The page number of the reference")

    @classmethod
    def from_document(cls, document: Document):
        return cls(
            content=document.page_content,
            title=document.metadata.get('source', 'Unknown Title'),
            page=str(document.metadata.get('page', 'Unknown Page'))
        )

@tool
def search_qdrant(query: Annotated[str, "The search query to find similar chunks"]) -> List[Reference]: 
    """
    Searches for similar chunks in a Qdrant collection based on a query.
    
    Args:
        query (str): The search query to find similar chunks.
        collection_name (str): The name of the Qdrant collection to search in.
        embeddings (AzureOpenAIEmbeddings): The embeddings model to use for the query.
    
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the search results with text and metadata.
    """
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text",
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"))
    
    docs = vector_store.similarity_search(query, k=5)
    refs = [Reference.from_document(doc) for doc in docs]
    return refs

checkpointer = MemorySaver()
prompt = ChatPromptTemplate.from_messages([
    ("system", open("prompt.txt", "r").read()),
    MessagesPlaceholder(variable_name="messages", optional=True)
])
tools = [search_qdrant]
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
    request_timeout=None,
    timeout=None,
    max_retries=3
).bind_tools(tools)

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "timeout": None,
        "recursion_limit": 10
    }
}

agent_executor = create_react_agent(llm, tools, state_modifier=prompt, state_schema=AgentState, checkpointer=checkpointer, debug=True)

ai_messages = []
state = {
    "messages": [HumanMessage(content="What is Domain Model and how could I implement it?")]
}

for s in agent_executor.stream(state, config, stream_mode="values"):
    if s["messages"]:
        message = s["messages"][-1]
        if isinstance(message, AIMessage):
            ai_messages.append(f"## AI Message\n\n{message.content}\n")
        elif isinstance(message, ToolMessage) and message.status == "success" and message.content:
            ai_messages.append(f"## Tool '{message.name}' Message\n\n{message.content}\n")


if ai_messages:
    write_to_file([x for x in ai_messages if x], f"data\\output\\{config["configurable"]["thread_id"][:6]}.md")