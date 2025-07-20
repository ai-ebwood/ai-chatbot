from langgraph.checkpoint.memory import MemorySaver
from rich import print as rprint
from datetime import datetime
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from configuration import Configuration
from state import State
load_dotenv(override=True)


async def call_llm(state: State, config: RunnableConfig) -> State:
    config: Configuration = Configuration.from_runnable_config(config)
    llm = init_chat_model(
        model=config.model,
        model_provider=config.model_provider
    )
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

agent_builder = StateGraph(State)
agent_builder.add_node("llm", call_llm)
agent_builder.add_edge(START, "llm")
agent_builder.add_edge("llm", END)

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
