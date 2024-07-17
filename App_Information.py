import streamlit as st
import asyncio

async def main():
    st.markdown("""
    <style>
    /* Centering title horizontally */
    .centered-title {
        text-align: center;
    }
    </style>""", unsafe_allow_html=True)

    st.title("LangGraph Cloud Query Processing App")

    st.write("This app demonstrates the use of LangGraph Cloud for processing complex queries. It's recommended to read through this guide to understand how it works.")

    st.header("Background")

    st.write("This app showcases LangGraph Cloud features in an interactive way. It processes user queries through a series of steps, including web search, planning, discovery, replanning, and generating a final response. The app uses multiple LLM models and a vector store to enhance the quality of responses.")

    st.header("The Graph Structure")

    st.write("The app uses a StateGraph to process queries. Here's an overview of the graph structure:")

    st.code("""
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", web_search)
    workflow.add_node("planner", planner)
    workflow.add_node("discovery", discovery)
    workflow.add_node("replanner", replanner)
    workflow.add_node("final_response", final_response)

    workflow.add_edge("web_search", "planner")
    workflow.add_edge("planner", "discovery")
    workflow.add_edge("discovery", "replanner")
    workflow.add_edge("replanner", "final_response")
    workflow.set_entry_point("web_search")
    workflow.add_edge("final_response", END)
    """, language="python")

    st.write("This graph structure shows the flow of data processing from web search to the final response.")

    st.header("Key Components")

    st.subheader("1. Vector Store")
    st.write("The app uses Chroma as a vector store with HuggingFaceEmbeddings. This allows for efficient storage and retrieval of relevant information.")

    st.subheader("2. LLM Models")
    st.write("Multiple LLM models are used for different tasks:")
    st.write("- planner_llm: mixtral-8x7b-32768")
    st.write("- collaborator_llm: llama3-70b-8192")
    st.write("- replanner_llm: gemma2-9b-it")
    st.write("- discovery_llm: gemma-7b-it")
    st.write("- final_response_llm: llama3-70b-8192")

    st.subheader("3. Web Search Tool")
    st.write("The app uses TavilySearchResults for web searches, enhancing the information available for query processing.")

    st.header("Processing Steps")

    st.write("1. **Web Search**: Searches the web for relevant information and adds it to the vector store.")
    st.write("2. **Planning**: Creates an initial plan to answer the user's question.")
    st.write("3. **Discovery**: Analyzes the plan and provides insights or improvements.")
    st.write("4. **Replanning**: Refines and optimizes the plan based on the insights.")
    st.write("5. **Final Response**: Generates a concise answer to the user's query based on the final plan.")

    st.header("Using the App")

    st.write("To use the app, simply enter your query when prompted. The app will process your query through the graph and provide a final answer.")

    st.header("Ideas for Future Work")

    st.write("Potential improvements for the app include:")
    st.write("- Enhancing the graph with more specialized nodes for different types of queries")
    st.write("- Adding visualizations of the graph processing steps")
    st.write("- Implementing user feedback mechanisms to improve responses over time")
    st.write("- Expanding the range of web search tools and information sources")
    st.write("- Optimizing the vector store for faster retrieval and more accurate results")
    st.write("- Adding support for multi-turn conversations and context retention")

    st.header("Technical Details")

    st.subheader("State Management")
    st.write("The app uses a GraphState TypedDict to manage the state throughout the processing:")
    st.code("""
    class GraphState(TypedDict):
        question: str
        web_results: List[str]
        plan: str
        insights: str
        final_plan: str
        final_answer: str
    """, language="python")

    st.subheader("Prompt Templates")
    st.write("The app uses several prompt templates for different stages of processing:")
    st.write("- Planner prompt: Guides the creation of an initial plan")
    st.write("- Discovery prompt: Analyzes the plan and provides insights")
    st.write("- Replanner prompt: Refines the plan based on insights")
    st.write("- Final response prompt: Generates the final answer")

    st.subheader("Vector Store Integration")
    st.write("The app integrates a Chroma vector store with HuggingFaceEmbeddings:")
    st.code("""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
    """, language="python")
    st.write("This allows for efficient storage and retrieval of relevant information throughout the query processing.")

    st.header("Performance Considerations")

    st.write("The app's performance can vary depending on several factors:")
    st.write("- Complexity of the user's query")
    st.write("- Amount of relevant information found during web search")
    st.write("- Response times of the various LLM models")
    st.write("- Size and efficiency of the vector store")

    st.write("To optimize performance, consider:")
    st.write("- Adjusting the number of web search results")
    st.write("- Fine-tuning the LLM models for specific tasks")
    st.write("- Implementing caching mechanisms for frequently asked queries")
    st.write("- Regularly maintaining and optimizing the vector store")

    st.header("Error Handling and Robustness")

    st.write("The app implements basic error handling and retries:")
    st.code("""
    ChatOpenAI(
        # ... other parameters ...
        timeout=None,
        max_retries=2,
    )
    """, language="python")
    st.write("Consider implementing more comprehensive error handling and logging for production use.")

    st.header("Conclusion")

    st.write("This LangGraph Cloud Query Processing App demonstrates the power of combining multiple LLM models, web search, and vector stores in a graph-based workflow. By processing queries through multiple stages of refinement, the app aims to provide high-quality, contextually relevant answers to user queries.")

    st.write("As you use the app, keep in mind that it's a prototype and may have limitations. Your feedback and experiences can help guide future improvements and optimizations.")

if __name__ == "__main__":
    asyncio.run(main())