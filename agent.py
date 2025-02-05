import os
import uuid
import json

from typing import List, Literal
from pydantic import BaseModel, Field

from langchain.schema import Document
from langchain.globals import set_verbose

from langchain_core.prompts import ChatPromptTemplate

from langchain_qdrant import QdrantVectorStore

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, END

set_verbose(True)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_MODEL"])

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
            id=document.metadata.get('_id', 'Unknown ID')
        )
    def __str__(self) -> str:
        """Returns a markdown formatted string representation of the reference"""
        return f"""
> {self.content}
>
> <a href="{self.id}">*{self.title}*, p. {self.page}</a>
"""

class RagState(AgentState):
    question: str
    queries: List[str]
    references: List[Reference]

checkpointer = MemorySaver()
tools = []
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
    request_timeout=None,
    timeout=None,
    max_retries=3
)
json_llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
    response_format={"type": "json_object"}
)

def generate_queries(state: RagState):
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\queries_prompt.txt", "r").read()),
        ("user", "{question}")])

    response = json_llm.invoke(prompt.format(question=question))
    parsed_response = json.loads(response.content)
    return {"queries": parsed_response["queries"]}

def retrieve_references(state: RagState):
    queries = state["queries"]
    retriever = QdrantVectorStore.from_existing_collection(
        collection_name=os.environ["QDRANT_COLLECTION"],
        embedding=embeddings,
        content_payload_key="text",
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY")).as_retriever(k=5)

    docs = [doc for query in queries for doc in retriever.invoke(query)]
    references = [Reference.from_document(doc) for doc in docs]
    unique_references = list({ref.id: ref for ref in references})
    return {"references": unique_references}

def generate_answer(state: RagState):
    query = state["question"]
    references_txt = "\n".join([str(ref) for ref in state["references"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\answer_prompt.txt", "r").read()),
        ("user", "{question}")])
    
    txt = prompt.format(question=query, references=references_txt)

    response = llm.invoke(txt)
    return {"messages": [{"role": "assistant", "content": response.content}]}

def evaluate_answer(state: RagState) -> Literal["end", "loop"]:
    answer = state["messages"][-1].content
    references = "\n".join([str(ref) for ref in state["references"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\grade_prompt.txt", "r").read()),
        ("user", "FACTS: {references}\n\nANSWER: {answer}")])
    
    result = json_llm.invoke(prompt.format(references=references, answer=answer))
    score = json.loads(result.content)["binary_score"]

    if score == "yes":
        return "end"
    else:
        return "loop"

if __name__ == "__main__":
    workflow = StateGraph(RagState)

    workflow.add_node("generatequeries", generate_queries);
    workflow.add_node("retrievereferences", retrieve_references);
    workflow.add_node("generateanswer", generate_answer);

    workflow.set_entry_point("generatequeries")
    workflow.add_edge("generatequeries", "retrievereferences")
    workflow.add_edge("retrievereferences", "generateanswer")
    workflow.add_conditional_edges(
        "generateanswer",
        evaluate_answer,
        { 
            "end": END,
            "loop": "generateanswer"
        }
    )

    # Compile
    graph = workflow.compile()
    graph.get_graph().draw_mermaid_png(output_file_path="data\\output\\workflow.png")

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "timeout": None,
            "recursion_limit": 10
        }
    }

    for event in graph.stream({"question": "What is Domain Model and how could i integrate it in my project ?"}, config=config):
        print(event)