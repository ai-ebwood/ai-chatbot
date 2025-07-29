# agent_service.py

import os
from datetime import datetime
import asyncio
from typing import TypedDict, List
from dotenv import load_dotenv

from langmem import create_memory_store_manager
from rich import print as rprint
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    MessageLikeRepresentation,
    HumanMessage,
    SystemMessage,
    AIMessage,
)

# --- 配置和类型定义 (保持不变) ---
load_dotenv()


class State(TypedDict):
    messages: list[MessageLikeRepresentation]

# --- AgentService 类 ---


class AgentService:
    def __init__(self, *, checkpointer: BaseCheckpointSaver, store: BaseStore):
        """
        初始化服务，建立数据库连接并编译 agent。
        这个方法在创建 AgentService 实例时只会被调用一次。
        """
        print("Initializing AgentService...")
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 1. 初始化数据库连接和存储
        # 我们不再使用 'with' 语句，而是将 checkpointer 和 store 保存为实例属性
        self.checkpointer = checkpointer
        self.store = store
        # 2. 初始化 langmem
        namespace_key = ("memories", "{user_id}")
        self.store_memory_manager = create_memory_store_manager(
            self.llm,
            namespace=namespace_key
        )

        # 3. 构建和编译 Graph
        agent_builder = StateGraph(State)
        agent_builder.add_node("llm", self._call_model)
        agent_builder.add_edge(START, "llm")
        agent_builder.add_edge("llm", END)

        self.agent = agent_builder.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
        print("AgentService initialized successfully.")

    def _call_model(self, state: State, config: RunnableConfig):
        """
        这是 Graph 中的节点，注意它现在是类的一个方法。
        """
        start_time = datetime.now()
        print(f"开始处理请求 (Thread ID: {config['configurable']['thread_id']})")

        messages = state["messages"]
        last_message = messages[-1]

        memories = self.store_memory_manager.search(
            query=last_message.content,
            config=config
        )
        rprint(f"当前记忆: {memories}")
        system_msg = f"""根据用户当前的相关记忆memories，回答用户的相关问题。

<memories>
{memories}
</memories>
"""
        response = self.llm.invoke(
            [SystemMessage(content=system_msg), *messages])

        all_messages = messages + [response]
        # 注意这里调用的是 self.store_memory_manager
        self.store_memory_manager.ainvoke({"messages": all_messages})
        total_time = datetime.now() - start_time
        print(f"完成处理请求 (Thread ID: {config['configurable']['thread_id']}), 花费时间: {total_time}")

        return {"messages": all_messages}

    async def arun(self, *, user_id: str, user_input: str) -> AIMessage:
        """
        这是暴露给外部调用的主方法。
        它封装了配置创建和调用 agent 的逻辑。
        """
        config = RunnableConfig(
            configurable={
                # 使用 user_id 创建一个唯一的 thread_id
                "thread_id": f"thread-for-{user_id}",
                "user_id": user_id
            }
        )

        # response = await self.agent.ainvoke(
        #     {"messages": [HumanMessage(content=user_input)]}, config=config
        # )
        response = await asyncio.to_thread(
            self.agent.invoke,
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        return response["messages"][-1]

    def run(self, *, user_id: str, user_input: str) -> AIMessage:
        """
        这是暴露给外部调用的主方法。
        它封装了配置创建和调用 agent 的逻辑。
        """
        config = RunnableConfig(
            configurable={
                # 使用 user_id 创建一个唯一的 thread_id
                "thread_id": f"thread-for-{user_id}",
                "user_id": user_id
            }
        )

        response = self.agent.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        return response["messages"][-1]

    # 你也可以把其他的辅助函数作为类的方法
    def get_state_history(self, user_id: str):
        config = RunnableConfig(
            configurable={
                "thread_id": f"thread-for-{user_id}",
                "user_id": user_id
            }
        )
        try:
            rprint(f"\n用户 '{user_id}' 的状态历史:\n")
            for state in self.agent.get_state_history(config=config):
                rprint(state.values.get("messages", []))
        except Exception as e:
            print(f"获取状态出错: {e}")
