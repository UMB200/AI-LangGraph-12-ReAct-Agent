from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import run_reasoning_agent_node, tool_node
load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1
app_msg = "What is the temperature in Tokyo? List it and then tripple it"

def should_continue(msg_state: MessagesState) -> str:
    if not msg_state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_reasoning_agent_node)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END, ACT:ACT})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")


if __name__ == "__main__":
    print("Running ReAct Langgraph with Function Calling")
    result = app.invoke({"messages": [HumanMessage(content=app_msg)]})
    print(result["messages"][LAST].content)


