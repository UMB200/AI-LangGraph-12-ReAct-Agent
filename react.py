from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
load_dotenv()

@tool
def triple_number(num:float) -> float:
    """
    param num: a number to triple
    return: the triple of the input
    """
    return float(num)*3

tool_to_use = [TavilySearch(max_result=1), triple_number]
llm_chatgpt = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tool_to_use)