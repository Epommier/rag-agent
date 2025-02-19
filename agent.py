import os
import uuid
import json

from embeddings_tools import get_vector_store
from typing import List, Literal, AsyncGenerator, Annotated
from pydantic import BaseModel, Field
from operator import add

from langchain.schema import Document, HumanMessage
from langchain.globals import set_verbose
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

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
            id=document.id
        )
    def __str__(self) -> str:
        """Returns a markdown formatted string representation of the reference"""
        return f"""
> *{self.content}*
>
>> **{self.title}**, p. {self.page}
"""

class Evaluation(BaseModel):
    binary_score: Literal["yes", "no"]
    explanation: str

class RagState(AgentState):
    question: str
    queries: List[str]
    references: List[Reference]
    evaluations: Annotated[List[Evaluation], add]

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment="gpt-4o",
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
    request_timeout=None,
    timeout=None,
    max_retries=3
)

json_llm = llm.bind(response_format={"type": "json_object"})

def generate_queries(state: RagState):
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\queries_prompt.txt", "r").read()),
        ("user", "{question}")])

    response = json_llm.invoke(prompt.format(question=question, language=os.environ["LANGUAGE"]))
    parsed_response = json.loads(response.content)
    return {
        "queries": parsed_response["queries"],
        "updates": [SystemMessage(content="📚 Searching through the document collection...")],
        "messages": [HumanMessage(content=question)]
    }

def retrieve_references(state: RagState):
    queries = state["queries"]

    vector_store = get_vector_store(os.environ["VECTOR_COLLECTION"], embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    docs = [doc for query in queries for doc in retriever.invoke(query)]
    references = [Reference.from_document(doc) for doc in docs]
    unique_refs_dict = {ref.id: ref for ref in references}

    return {
        "references": list(unique_refs_dict.values()),
        "updates": [
            SystemMessage(content=f"📎 Found {len(unique_refs_dict)} relevant references."),
        ]
    }

def generate_answer(state: RagState):
    query = state["question"]
    references_txt = "\n\n".join([str(ref) for ref in state["references"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\answer_prompt.txt", "r").read()),
        ("user", "REFERENCES: {references}\n\nQUESTION: {question}")])

    response = llm.invoke(
        prompt.format(
            language=os.environ["LANGUAGE"],
            question=query,
            references=references_txt
        )
    )
    
    return {
        "messages": [AIMessage(content=response.content)]
    }

def evaluate_answer(state: RagState):
    answer = state["messages"][-1].content
    references = "\n".join([str(ref) for ref in state["references"]])
    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\grade_prompt.txt", "r").read()),
        ("user", "FACTS: {references}\n\nANSWER: {answer}")]
    )
    
    result = json_llm.invoke(prompt.format(references=references, answer=answer))
    evaluation = Evaluation.model_validate_json(result.content)

    return {
        "evaluations": [evaluation]
    }

def evaluate_edge(state: RagState) -> Literal["end", "loop"]:
    evaluation = state["evaluations"][-1]
    if evaluation.binary_score == "yes":
        return "end"
    else:
        return "loop"

def correct_answer(state: RagState):
    answer = state["messages"][-1].content
    evaluation = state["evaluations"][-1]
    question = state["question"]
    references = "\n".join([str(ref) for ref in state["references"]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", open("prompts\\grade_prompt.txt", "r").read()),
        ("user", """QUESTION: {question}\n\n
            REFERENCES: {references}\n\n
            Improve the below answer, taking into account the professor's remarks.\n\n
            ANSWER: {answer}\n\n
            PROFESSOR REMARKS:{evaluation}"""
        )
    ])

    response = llm.invoke(prompt.format(
        references=references,
        question=question,
        answer=answer,
        evaluation=evaluation.justification)
    )

    return {
        "messages": [AIMessage(content=response.content)]
    }

def construct_agent_graph() -> CompiledStateGraph:
    checkpointer = MemorySaver()
    workflow = StateGraph(RagState)

    workflow.add_node("generatequeries", generate_queries);
    workflow.add_node("retrievereferences", retrieve_references);
    workflow.add_node("generateanswer", generate_answer);
    workflow.add_node("evaluateanswer", evaluate_answer);
    workflow.add_node("correctanswer", correct_answer);

    workflow.set_entry_point("generatequeries")
    workflow.add_edge("generatequeries", "retrievereferences")
    workflow.add_edge("retrievereferences", "generateanswer")
    workflow.add_edge("generateanswer", "evaluateanswer")
    workflow.add_edge("correctanswer", "evaluateanswer")
    workflow.add_conditional_edges(
        "evaluateanswer",
        evaluate_edge,
        { 
            "end": END,
            "loop": "correctanswer"
        }
    )

    # Compile
    graph = workflow.compile(checkpointer=checkpointer)
    return graph

async def arun_agent(question: str) -> AsyncGenerator[dict, None]:
    """Run the agent asynchronously and yield intermediate results"""
    graph = construct_agent_graph()    
    config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "timeout": None,
                "recursion_limit": 10
            }
        }

    # Create initial state
    initial_state = RagState(
        question=question,
        queries=[],
        references=[]
    )

    async for _ in graph.astream(initial_state, config=config):
        data = graph.get_state(config=config).values
        yield data

if __name__ == "__main__":
    graph = construct_agent_graph()
    graph.get_graph().draw_mermaid_png(output_file_path="data\\output\\workflow.png")