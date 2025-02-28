import os
from langgraph.graph import Graph
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda

# Ensure API keys are set
if "TAVILY_API_KEY" not in os.environ:
    raise ValueError("TAVILY_API_KEY not found. Set it as an environment variable.")

# Initialize Tavily search tool
search_tool = TavilySearchResults()

def research_agent(inputs):
    """Agent to perform research using Tavily."""
    query = inputs["query"]  # Extract query from inputs
    results = search_tool.invoke(query)

    if not results:
        print("[ERROR] No research results found!")
        return {"query": query, "research_results": "No relevant results found."}

    research_results = "\n\n".join([res.get("content", "No content available") for res in results])
    print("\n[DEBUG] Research Results:\n", research_results)
    return {"query": query, "research_results": research_results}  

# Initialize Ollama LLM
llm = OllamaLLM(model="mistral")
# Define answer drafting template
answer_prompt = PromptTemplate(
    input_variables=["query", "research_results"],
    template="""
    Based on the research results below, draft a detailed and well-structured response to the query:
    Query: {query}
    Research Results:
    {research_results}
    """
)
def answer_drafting_agent(inputs):
    """Agent to generate an answer based on research results."""
    query = inputs["query"]
    research_results = inputs["research_results"]

    if not research_results.strip():
        print("[ERROR] No research results provided to the answer drafting agent!")
        return {"drafted_answer": "Unable to generate an answer due to lack of research data."}

    prompt = answer_prompt.format(query=query, research_results=research_results)
    
    print("\n[DEBUG] Sending prompt to LLM:\n", prompt)  # Debugging step

    try:
        response = llm.invoke(prompt)
        print("\n[DEBUG] LLM Raw Response:\n", response)  # Debugging step

        if response is None:
            print("[ERROR] LLM response is None!")
            return {"drafted_answer": "Failed to generate an answer."}

        return {"drafted_answer": response}

    except Exception as e:
        print(f"[ERROR] LLM invocation failed: {e}")
        return {"drafted_answer": "An error occurred during answer generation."}

# Create a LangGraph workflow
graph = Graph()
graph.add_node("research_agent", RunnableLambda(research_agent))
graph.add_node("answer_drafting_agent", RunnableLambda(answer_drafting_agent))

graph.add_edge("research_agent", "answer_drafting_agent")
graph.set_entry_point("research_agent")

workflow = graph.compile()
def main():
    query = "What are the latest advancements in artificial intelligence?"
    result = workflow.invoke({"query": query})

    print("\n[DEBUG] Workflow Output:\n", result)  # Ensure it contains the expected keys

    if "drafted_answer" not in result or not result["drafted_answer"]:
        print("[ERROR] Answer drafting failed. No valid answer returned.")
        return

    print("\nGenerated Answer:\n", result["drafted_answer"])


if __name__ == "__main__":
    main()



