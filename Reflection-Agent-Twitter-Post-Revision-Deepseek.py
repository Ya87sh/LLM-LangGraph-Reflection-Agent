# %%
from langgraph import Graph, Node
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize the LLM (GPT-4)
llm = ChatOpenAI(model="gpt-4")

# Define the Generate Node
def generate_node(state):
    
    original_tweet = state.get("original_tweet", "")

    # Generate a new tweet based on the original tweet
    response = llm([
        SystemMessage(content="You are a helpful assistant that generates engaging Twitter posts about LangChain."),
        HumanMessage(content=f"Generate a Twitter post based on this: {original_tweet}")
    ])
    
    new_tweet = response.content
    
    state["current_tweet"] = new_tweet
    
    return state

# Define the Revise Node
def revise_node(state):
    
    current_tweet = state.get("current_tweet", "")
    
    # Critique and revise the current tweet
    response = llm([
        SystemMessage(content="You are a critical assistant that revises Twitter posts to make them more engaging and likely to go viral."),
        HumanMessage(content=f"Revise this Twitter post to improve it: {current_tweet}")
    ])
    
    revised_tweet = response.content
    state["current_tweet"] = revised_tweet
    state["iteration"] = state.get("iteration", 0) + 1
    
    return state

# Create the graph
workflow = Graph()

# Add nodes to the graph
workflow.add_node("generate", generate_node)
workflow.add_node("revise", revise_node)

# Define edges
workflow.add_edge("generate", "revise")
workflow.add_edge("revise", "generate")

# Define the entry point
workflow.set_entry_point("generate")

# Define the condition to continue iterating
def continue_condition(state):
    return state.get("iteration", 0) < 6

# Add a conditional edge
workflow.add_conditional_edge("revise", continue_condition, "generate")

# Compile the graph
app = workflow.compile()

# Function to run the reflection agent
def run_reflection_agent(original_tweet):
    state = {"original_tweet": original_tweet, "iteration": 0}
    final_state = app.invoke(state)
    return final_state["current_tweet"]

# Example usage
original_tweet = "LangChain is a framework for developing applications powered by language models."
final_tweet = run_reflection_agent(original_tweet)
print("Final Tweet:", final_tweet)

# %%



