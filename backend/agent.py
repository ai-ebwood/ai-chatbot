from typing import Literal
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from rich import print as rprint
from datetime import datetime
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from configuration import Configuration
from state import State
from tools import (
    ddg_search
)
from dotenv import load_dotenv
load_dotenv(override=True)

tools = [ddg_search]
tool_by_name = {tool.name: tool for tool in tools}


async def call_llm(state: State, config: RunnableConfig) -> State:
    """A node to call llm."""
    config: Configuration = Configuration.from_runnable_config(config)
    llm = init_chat_model(
        model=config.model,
        model_provider=config.model_provider
    ).bind_tools(tools)

    last_message = state["messages"][-1]
    total_tokens = state["total_tokens"]
    new_tokens = len(last_message.content) * 3
    # rprint(f"消息大小: {len(state["messages"])}, 状态: {state}")
    if total_tokens + new_tokens >= config.max_token_limit:
        rprint(
            f"超出了token大小: limit_token = {config.max_token_limit}, new_total_tokens = {total_tokens + new_tokens}")
        return {"exceed_tokens": True}

    response = await llm.ainvoke(state["messages"])
    total_tokens = response.usage_metadata["total_tokens"]

    return {"messages": [response], "total_tokens": total_tokens, "exceed_tokens": False}


def should_continue(state: State) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return END


def tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    result = []
    for tool_call in tool_calls:
        tool_result = tool_by_name[tool_call["name"]].invoke(tool_call["args"])
        result.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": result}


agent_builder = StateGraph(State)
agent_builder.add_node("llm", call_llm)
agent_builder.add_node("tools", tool_node)
agent_builder.add_edge(START, "llm")
agent_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
agent_builder.add_edge("tools", "llm")

checkpointer = MemorySaver()


async def run_agent(*, model: str = "gpt-4o-mini", model_provider: str = "openai", question: str, user_id: str | None = None):
    user_id = user_id or datetime.now().timestamp()
    config: RunnableConfig = {"configurable": {
        "thread_id": user_id,
        "user_id": user_id,
        "model": model,
        "model_provider": model_provider
    }}
    agent = agent_builder.compile(
        checkpointer=checkpointer
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config=config)
    if result["exceed_tokens"]:
        return {"content": "超出token大小，请开始新聊天"}

    data = result["messages"][-1]
    rprint(data)
    return data


async def main():
    agent = agent_builder.compile()
    # async for chunk in agent.astream({"messages": [HumanMessage(content="特斯拉2025年股价趋势")]}, stream_mode="values"):
    #     chunk["messages"][-1].pretty_print()
    result = await run_agent(question="特斯拉2025年股价趋势")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
