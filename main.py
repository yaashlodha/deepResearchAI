import os
from langgraph.graph import Graph
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda


if "TAVILY_API_KEY" not in os.environ:
    raise ValueError("TAVILY_API_KEY not found.")


search_tool = TavilySearchResults()

def research_agent(inputs):
    
    query = inputs.get("query", "")
    print(f"\nresearch_agent received query: {query}")

    if not query.strip():
        print("research_agent received an empty query.")
        return {"query": query, "research_results": "Invalid query."}

    try:
        results = search_tool.invoke(query)
    except Exception as e:
        print(f"Tavily API call failed: {e}")
        return {"query": query, "research_results": "Error retrieving research results."}

    if not results or not isinstance(results, list):
        print("No research results found!")
        return {"query": query, "research_results": "No relevant results found."}

    research_results = "\n\n".join([res.get("content", "No content available") for res in results])
    print(f"\nresearch_agent returning results:\n{research_results[:500]}")  # Limit output size
    
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
    print(f"\n[DEBUG] answer_drafting_agent received inputs:\n{inputs}")

    query = inputs.get("query", "")
    research_results = inputs.get("research_results", "")

    if not research_results.strip():
        print("No research results provided to the answer drafting agent!")
        return {"drafted_answer": "Unable to generate an answer due to lack of research data."}

    prompt = answer_prompt.format(query=query, research_results=research_results)
    
    print("\nSending prompt to LLM:\n", prompt[:500])  # Limit output size

    try:
        response = llm.invoke(prompt)
        print("\nLLM Raw Response:\n", response[:500])  # Limit output size

        if response is None or not isinstance(response, str):
            print("LLM response is invalid!")
            return {"drafted_answer": "Failed to generate an answer."}

        return {"drafted_answer": response.strip()}  # Ensure clean output

    except Exception as e:
        print(f"LLM invocation failed: {e}")
        return {"drafted_answer": "An error occurred during answer generation."}


# Create a LangGraph workfow
graph = Graph()

# Debugging wrapper to track node execution
def debug_wrapper(func, name):
    def wrapped(inputs):
        print(f"\n[DEBUG] Executing {name} with inputs:\n{inputs}")
        result = func(inputs)
        print(f"\n[DEBUG] Output from {name}:\n{result}")
        return result
    return wrapped

graph.add_node("research_agent", RunnableLambda(debug_wrapper(research_agent, "research_agent")))
graph.add_node("answer_drafting_agent", RunnableLambda(debug_wrapper(answer_drafting_agent, "answer_drafting_agent")))

graph.add_edge("research_agent", "answer_drafting_agent")

# Set the final output node
graph.set_entry_point("research_agent")
graph.set_finish_point("answer_drafting_agent")  # ✅ Explicitly tell LangGraph where to stop

workflow = graph.compile()


def main():
    query = input("Enter your query: ")
    result = workflow.invoke({"query": query})
    f=open("answer.txt","w")
    f.write(result["drafted_answer"])
    f.close()




if __name__ == "__main__":
    main()


