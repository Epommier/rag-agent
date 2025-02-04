import os
import uuid
import json

from typing import Annotated, List
from pydantic import BaseModel, Field
from slugify import slugify

from langchain.schema import Document
from langchain.globals import set_verbose
from langchain.prompts import MessagesPlaceholder
from langchain.tools import tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from langchain_qdrant import QdrantVectorStore

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState

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
    id: str = Field(..., description="The unique identifier of the reference")

    @classmethod
    def from_document(cls, document: Document):
        return cls(
            content=document.page_content,
            title=document.metadata.get('source', 'Unknown Title'),
            page=str(document.metadata.get('page', 'Unknown Page')),
            id=document.id)
    
    def __str__(self) -> str:
        """Returns a markdown formatted string representation of the reference"""
        return f"""
> {self.content}
>
> <a href="{slugify(self.tile)}-{self.page}">*{self.title}*, p. {self.page}</a>
"""

class RagState(AgentState):
    question: str
    queries: List[str]
    references: List[Reference]

@tool
def search_qdrant(query: Annotated[str, "The search query to find similar text in documents"]) -> List[Reference]: 
    """
    This function performs a semantic search in a document collection to find text chunks in documents that are similar to the given query.
    
    Args:
        query (str): The search query to find similar text in documents.
    
    Returns:
        List[Reference]: A list of references containing the search results with text, title and page.
    """

    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text",
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY")).as_retriever(k=5)
    
    docs = retriever.get_relevant_documents(query)
    refs = [Reference.from_document(doc) for doc in docs]
    return refs

checkpointer = MemorySaver()
tools = [search_qdrant]
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
    request_timeout=None,
    timeout=None,
    max_retries=3
)

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4()),
        "timeout": None,
        "recursion_limit": 10
    }
}

def generate_question(state: RagState):
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\queries_prompt.txt", "r").read()),
        ("user", "{question}")])

    json_llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment="gpt-4o",
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=0.0,
        response_format={"type": "json_object"})

    response = json_llm.invoke(prompt.format(question=question))
    parsed_response = json.loads(response.content)
    return {"queries": parsed_response["queries"]}

def retrieve_references(state: RagState):
    queries = state["queries"]
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text",
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY")).as_retriever(k=5)

    docs = [doc for query in queries for doc in retriever.invoke(query)]
    return {"references": [Reference.from_document(doc) for doc in docs]}

def generate_answer(state: RagState):
    query = state["question"]
    references_txt = "\n".join([str(ref) for ref in state["references"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\answer_prompt.txt", "r").read()),
        ("user", "{question}")])
    
    txt = prompt.format(question=query, references=references_txt)

    response = llm.invoke(txt)
    return {"messages": [{"role": "assistant", "content": response.content}]}