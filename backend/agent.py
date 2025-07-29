from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from typing import TypedDict
from typing import Literal
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from rich import print as rprint
from datetime import datetime
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.messages import trim_messages
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langgraph.config import get_store, get_config
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import get_buffer_string
from langmem.short_term import summarize_messages, SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
from prompts import (
    generate_result_prompt,
    generate_summary_prompt)

from configuration import Configuration
from state import State, SummaryOutput
from tools import (ddg_search)
from langchain_qdrant import QdrantVectorStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_postgres import PGVectorStore, PGVector, PGVectorTranslator, PostgresChatMessageHistory
from langmem import create_manage_memory_tool, create_search_memory_tool, create_prompt_optimizer

from dotenv import load_dotenv
load_dotenv(override=True)

tools = [ddg_search]
tool_by_name = {tool.name: tool for tool in tools}

def get_llm(model: str, model_provider: str) -> BaseChatModel:
    return init_chat_model(
        model=model,
        model_provider=model_provider
    )


def get_summary(config: RunnableConfig, store: BaseStore):
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("user_id in config should set")
    namespace = ("data", )
    summary = store.get(namespace, user_id)
    # rprint(f"总结: {summary}")
    if summary is None:
        return ""
    return summary.value["summary"]


summarization_model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai"
)

summarization_node = SummarizationNode(
    model=summarization_model,
    # token_counter=count_tokens_approximately,
    token_counter=ChatOpenAI(
        model="gpt-4o-mini").get_num_tokens_from_messages,
    max_tokens=60,
    max_tokens_before_summary=60,
    max_summary_tokens=40
)


class MyState(State):
    context: dict[str, RunningSummary]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]


async def call_model(state: LLMInputState, config: RunnableConfig, store: BaseStore):
    custom_config: Configuration = Configuration.from_runnable_config(config)
    llm = get_llm(
        model=custom_config.model,
        model_provider=custom_config.model_provider
    ).bind_tools(tools)

    rprint("="*50 + "开始" + "="*50)
    rprint(f"请求消息大小: {len(state["summarized_messages"])}")
    rprint(f"summarized_messages: {state["summarized_messages"]}")
    response = await llm.ainvoke(state["summarized_messages"])

    rprint("\n\n响应:")
    rprint(f"{response}")
    rprint("="*50 + "结束" + "="*50)

    return {"messages": [response]}


def save_summary(config: RunnableConfig, store: BaseStore, summary: str):
    user_id = config["configurable"]["user_id"]
    if user_id is None:
        raise ValueError("user_id in config should set")
    namespace = ("data", )
    store.put(namespace, user_id, {"summary": summary})


async def call_llm(state: State, config: RunnableConfig) -> State:
    """A node to call llm."""
    custom_config: Configuration = Configuration.from_runnable_config(config)
    llm = get_llm(
        model=custom_config.model,
        model_provider=custom_config.model_provider
    ).bind_tools(tools)

    all_messages = state["messages"]

    # only save last 4 messages
    # all_messages = all_messages[-4:]

    # trim message by token count
    # all_messages = trim_messages(
    #     all_messages,
    #     max_tokens=40,
    #     token_counter=ChatOpenAI(
    #         model="gpt-4o-mini"
    #     )
    # )

    last_message = state["messages"][-1]
    total_tokens = state["total_tokens"]
    new_tokens = len(last_message.content) * 3
    rprint(
        f"消息大小: {len(all_messages)}, id = {last_message.id}, total_tokens = {state["total_tokens"]}")
    if total_tokens + new_tokens >= custom_config.max_token_limit:
        rprint(
            f"超出了token大小: limit_token = {custom_config.max_token_limit}, new_total_tokens = {total_tokens + new_tokens}")
        return {"exceed_tokens": True}

    summary = get_summary(config, store)
    response = await llm.ainvoke([
        SystemMessage(
            content=generate_result_prompt.format(
                summary=summary
            )
        ),
        last_message
    ])
    total_tokens = response.usage_metadata["total_tokens"]

    return {"messages": [response], "total_tokens": total_tokens, "exceed_tokens": False}


# def should_continue(state: State) -> Literal["tools", "summarize"]:
#     if state["messages"][-1].tool_calls:
#         return "tools"
#     return "summarize"
def should_continue(state: State) -> Literal["tools", "__end__"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "__end__"


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


async def summarize(state: State, config: RunnableConfig, store: BaseStore):
    """get messages summary"""
    custom_config: Configuration = Configuration.from_runnable_config(config)
    summary = get_summary(config, store)

    last_human_message_index = -1
    all_messages = state["messages"]
    for i in reversed(range(len(all_messages))):
        message = all_messages[i]
        if isinstance(message, HumanMessage):
            last_human_message_index = i
            break

    need_summary_messages = all_messages[last_human_message_index:]
    if len(need_summary_messages) == 0:
        return

    user_prompt = ChatPromptTemplate.from_template(
        generate_summary_prompt
    )

    llm = get_llm(
        model=custom_config.model,
        model_provider=custom_config.model_provider
    )
    chain = user_prompt | llm.with_structured_output(SummaryOutput)

    response = chain.invoke({
        "summary": summary,
        "messages": get_buffer_string(need_summary_messages)
    })
    save_summary(config, store, response["summary"])


agent_builder = StateGraph(MyState)
# agent_builder.add_node("llm", call_llm)
agent_builder.add_node("llm", call_model)
agent_builder.add_node("tools", tool_node)
# agent_builder.add_node("summarize", summarize)
# agent_builder.add_edge(START, "llm")
agent_builder.add_node("summarize", summarization_node)
agent_builder.add_edge(START, "summarize")
agent_builder.add_edge("summarize", "llm")
agent_builder.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
agent_builder.add_edge("tools", "llm")
# agent_builder.add_edge("summarize", END)

checkpointer = MemorySaver()
store = InMemoryStore()


async def run_agent(*, model: str = "gpt-4o-mini", model_provider: str = "openai", question: str, user_id: str | None = None):
    user_id = user_id or datetime.now().timestamp()
    config: RunnableConfig = {"configurable": {
        "thread_id": user_id,
        "user_id": user_id,
        "model": model,
        "model_provider": model_provider
    }}
    agent = agent_builder.compile(
        checkpointer=checkpointer,
        store=store
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]}, config=config)
    if result["exceed_tokens"]:
        return {"content": "超出token大小，请开始新聊天"}

    data = result["messages"][-1]
    rprint(data)
    return data


async def main():
    agent = agent_builder.compile(checkpointer=checkpointer, store=store)
    # async for chunk in agent.astream({"messages": [HumanMessage(content="特斯拉2025年股价趋势")]}, stream_mode="values"):
    #     chunk["messages"][-1].pretty_print()
    user_id = datetime.now().timestamp()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-123",
            "user_id": user_id
        }
    }
    while True:
        user_input = input("Your: ")
        # if user_input.lower() == "summary":
        #     summary = get_summary(config, store)
        #     rprint(f"当前总结: {summary}")
        #     continue
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config)
        # print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
