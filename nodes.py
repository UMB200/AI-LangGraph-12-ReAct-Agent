from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import llm_chatgpt, tool_to_use

load_dotenv()

SYSTEM_MESSAGE="""
You are a helpful assistant that can use tools to answer questions.
"""

def run_reasoning_agent_node(state:MessagesState)-> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm_chatgpt.invoke([
        {"role" : "system",
        "content" : SYSTEM_MESSAGE},
        *state["messages"]])
    return {"messages": [response]}

tool_node = ToolNode(tool_to_use)
