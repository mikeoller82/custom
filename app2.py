import os
from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rich.markdown import Markdown
from langchain_huggingface import HuggingFaceEmbeddings
from rich.panel import Panel
from rich.console import Console
from rich import print
from copy import deepcopy
from dotenv import load_dotenv

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

# Set API keys
os.getenv("GROQ_API_KEY")
os.getenv("OPENAI_API_KEY")
os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.getenv("TAVILY_API_KEY")

# Initialize HuggingFaceEmbeddings
console.print("[bold yellow]Initializing HuggingFaceEmbeddings...[/bold yellow]")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
console.print("[bold green]HuggingFaceEmbeddings initialized successfully![/bold green]")

# Initialize vector store with HuggingFaceEmbeddings
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

# Create LLM instances for different nodes
def create_llm(model_name, api_key):
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

planner_llm = create_llm("mixtral-8x7b-32768", metadata={"name":"planner_llm"}, api_key=os.getenv("GROQ_API_KEY"))
collaborator_llm = create_llm("llama3-70b-8192", metadata={"name":"collaborator_llm"}, api_key=os.getenv("GROQ_API_KEY"))
replanner_llm = create_llm("gemma2-9b-it", metadata={"name":"replanner_llm"}, api_key=os.getenv("GROQ_API_KEY"))
discovery_llm = create_llm("gemma-7b-it", metadata={"name":"discovery_llm"}, api_key=os.getenv("GROQ_API_KEY"))
final_response_llm = create_llm("llama3-70b-8192", metadata={"name":"final_response_llm"}, api_key=os.getenv("GROQ_API_KEY"))


# Web Search Tool
web_search_tool = TavilySearchResults(k=3)

# Function to print markdown
def print_md(text):
    console.print(Markdown(text))
    
# Function to add data to vector store
def add_to_vector_store(text: str, metadata: Dict[str, Any] = {}):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    
    docs = [Document(page_content=split, metadata=metadata) for split in splits]
    vectorstore.add_documents(docs)

# Function to load web content
def load_web_content(url: str) -> str:
    loader = WebBaseLoader(url)
    data = loader.load()
    return data[0].page_content if data else ""

# Prompt templates
planner_messages = [
    ("system", "You are a strategic planner. Create a plan to answer the user's question using the provided tools."),
    ("human", "Create a plan to answer this question: {question}\nConsider these web search results: {web_results}")
]

discovery_messages = [
    ("system", "You are a discovery agent. Analyze the plan and provide insights or improvements."),
    ("human", "Analyze this plan and provide insights or improvements for the question '{question}':\n{plan}")
]

replanner_messages = [
    ("system", "You are a replanning expert. Refine and optimize the plan based on the provided insights."),
    ("human", "Refine and optimize the plan for the question '{question}' based on these insights:\n{insights}")
]

final_response_messages = [
    ("system", "You are the final response generator. Summarize the process and provide a concise answer to the user's query."),
    ("human", "Provide a final answer to this question based on the final plan:\nQuestion: {question}\nFinal Plan: {final_plan}")
]

planner_prompt = ChatPromptTemplate.from_messages(planner_messages)
discovery_prompt = ChatPromptTemplate.from_messages(discovery_messages)
replanner_prompt = ChatPromptTemplate.from_messages(replanner_messages)
final_response_prompt = ChatPromptTemplate.from_messages(final_response_messages)

planner_chain = planner_prompt | planner_llm | StrOutputParser()
discovery_chain = discovery_prompt | discovery_llm | StrOutputParser()
replanner_chain = replanner_prompt | replanner_llm | StrOutputParser()
final_response_chain = final_response_prompt | final_response_llm | StrOutputParser()

# State Definitions
class GraphState(TypedDict):
    question: str
    web_results: List[str]
    plan: str
    insights: str
    final_plan: str
    final_answer: str

def web_search(state: GraphState) -> GraphState:
    print_md("## Web Search")
    console.print(Panel("[bold cyan]WEB SEARCH[/bold cyan]", expand=False))
    question = state["question"]
    results = web_search_tool.invoke({"query": question})
    web_results = []
    for result in results:
        content = result["content"]
        url = result.get("url", "")
        if url:
            full_content = load_web_content(url)
            add_to_vector_store(full_content, {"source": "web_search", "url": url})
        else:
            add_to_vector_store(content, {"source": "web_search"})
        web_results.append(content)
    
    return {**state, "web_results": web_results}

def planner(state: GraphState) -> GraphState:
    plan = planner_chain.invoke({
        "question": state['question'],
        "web_results": state['web_results']
    })
    return {**state, "plan": plan}

def discovery(state: GraphState) -> GraphState:
    insights = discovery_chain.invoke({
        "question": state['question'],
        "plan": state['plan']
    })
    return {**state, "insights": insights}

def replanner(state: GraphState) -> GraphState:
    final_plan = replanner_chain.invoke({
        "question": state['question'],
        "insights": state['insights']
    })
    return {**state, "final_plan": final_plan}

def final_response(state: GraphState) -> GraphState:
    final_answer = final_response_chain.invoke({
        "question": state['question'],
        "final_plan": state['final_plan']
    })
    return {**state, "final_answer": final_answer}

# Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("planner", planner)
workflow.add_node("discovery", discovery)
workflow.add_node("replanner", replanner)
workflow.add_node("final_response", final_response)

# Define edges
workflow.add_edge("web_search", "planner")
workflow.add_edge("planner", "discovery")
workflow.add_edge("discovery", "replanner")
workflow.add_edge("replanner", "final_response")
workflow.set_entry_point("web_search")
workflow.add_edge("final_response", END)

# Compile the graph
app = workflow.compile()

# Function to run the graph
def run_graph(query: str):
    initial_state = {"question": query}
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Node '{key}' completed")
            print(f"Current State:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        print("\n---\n")
    print("Final Answer:", output.get("final_answer", "No final answer generated."))

# Main execution
if __name__ == "__main__":
    user_query = input("Enter your task or query: ")
    run_graph(user_query)