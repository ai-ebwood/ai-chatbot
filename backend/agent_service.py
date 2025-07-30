# agent_service.py
from typing import Coroutine
from chat_history import PostgresChatMessageHistoryWithId
from state import (
    AgentState,
    UserProfile
)
import uuid
from langchain_core.vectorstores.base import VectorStore
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from prompts import (
    system_prompt,
    generate_memory_prompt)
from langchain_core.documents import Document
from langmem import create_memory_store_manager
from langmem.utils import get_conversation
from rich import print as rprint
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)
from langchain_core.prompts import SystemMessagePromptTemplate
import psycopg
from constants import CONVERSATION_TABLE_NAME

# --- 配置和类型定义 (保持不变) ---
load_dotenv()


class AgentService:
    # --- AgentService 类 ---
    def __init__(
        self,
        *,
        checkpointer: BaseCheckpointSaver,
        store: BaseStore,
        vector_store: VectorStore,
        db_conn_async: psycopg.AsyncConnection
    ):
        """
        初始化服务，建立数据库连接并编译 agent。
        这个方法在创建 AgentService 实例时只会被调用一次。
        """
        print("Initializing AgentService...")
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.checkpointer = checkpointer
        self.store = store
        self.vector_store = vector_store
        self.db_conn_async = db_conn_async

        # 2. 初始化 langmem
        self.namespace_key = ("memories", "{user_id}")
        self.store_memory_manager = create_memory_store_manager(
            self.llm,
            namespace=self.namespace_key,
            schemas=[UserProfile],
            instructions=generate_memory_prompt,
            enable_deletes=True
        )

        # 3. 构建和编译 Graph
        agent_builder = StateGraph(AgentState)
        agent_builder.add_node("llm", self._call_model)
        agent_builder.add_edge(START, "llm")
        agent_builder.add_edge("llm", END)

        self.agent = agent_builder.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
        print("AgentService initialized successfully.")

    def _get_db(self, conversation_id: str):
        db = PostgresChatMessageHistoryWithId(
            CONVERSATION_TABLE_NAME,
            conversation_id,
            async_connection=self.db_conn_async)
        return db

    def _filter_messages(self, state: AgentState):
        current_round_messages = []
        all_messages = state["messages"]
        for message in reversed(all_messages):
            current_round_messages.append(message)
            if isinstance(message, HumanMessage):
                break
        return current_round_messages

    def _create_config(self, user_id: str, conversation_id: str) -> RunnableConfig:
        return RunnableConfig(
            configurable={
                # 使用 user_id 创建一个唯一的 thread_id
                "thread_id": f"thread-for-{conversation_id}",
                "user_id": user_id,
                "conversation_id": conversation_id
            }
        )

    async def _save_memory(self, messages, config):
        """
        异步保存记忆。
        修正了对 _save_conversation 的调用，正确传递 user_id。
        """
        # 从 config 中获取 user_id
        user_id = config["configurable"]["user_id"]
        conversation_id = config["configurable"]["conversation_id"]

        # 保存提取的信息
        save_store_task = asyncio.create_task(
            self._save_store(messages, config))
        # 保存本轮消息的向量
        save_vector_conversation_task = asyncio.create_task(
            self._save_vector_conversation(messages, user_id, conversation_id))
        # 保存本轮消息列表
        save_conversation_task = asyncio.create_task(
            self._save_conversation(messages, user_id, conversation_id))

        # 等待两个任务完成 (可选，但推荐用于更好的错误处理)
        await asyncio.gather(
            save_store_task,
            save_vector_conversation_task,
            save_conversation_task)

    async def _save_store(self, messages: list[BaseMessage], config):
        start_time = datetime.now()
        rprint(f"后台保存记忆开始: {start_time}")
        save_store_result = await self.store_memory_manager.ainvoke(
            {"messages": messages}, config=config)
        cost_time = datetime.now() - start_time
        rprint(
            f"后台保存记忆成功: {datetime.now()}, cost_time: {cost_time}\n{save_store_result}")

    async def _save_vector_conversation(self, messages: list[BaseMessage], user_id: str, conversation_id: str):
        start_time = datetime.now()
        rprint(f"后台插入vector会话开始: {start_time}")
        # 只保存Human + AI 消息作为向量
        page_content = "\n".join(
            [f"{message.type}: {message.content}" for message in messages if message.type in ['human', 'ai']])
        # 这个轮次的消息则包括ToolMessage等
        message_ids = [str(msg.id) for msg in messages]

        doc = Document(page_content=page_content, metadata={
            "message_ids": message_ids,
            "user_id": user_id,
            "conversation_id": conversation_id
        })
        doc_id = str(uuid.uuid4())
        try:
            result = await self.vector_store.aadd_documents([doc], ids=[doc_id])
            cost_time = datetime.now() - start_time
            rprint(
                f"后台插入vector会话成功: {datetime.now()}, cost_time: {cost_time}, id = {doc_id}, result: \n{result}")
        except Exception as e:
            rprint(f"[bold red]后台插入会话失败: {e}[/bold red]")
            # 可以在这里进一步打印 doc 的内容以供调试
            # rprint(f"Failed to save document: {doc}")

    async def _save_conversation(self, messages: list[BaseMessage], user_id: str, conversation_id: str):
        start_time = datetime.now()
        rprint(f"后台插入消息列表开始: {start_time}\n{messages}")
        db = self._get_db(conversation_id)
        try:
            result = await db.aadd_messages(messages)
            cost_time = datetime.now() - start_time
            rprint(
                f"后台插入消息列表成功: {datetime.now()}, cost_time: {cost_time}, result: {result}")
        except Exception as e:
            rprint(
                f"[bold red]插入{CONVERSATION_TABLE_NAME}消息列表失败: {datetime.now()}, end_time: {datetime.now()}\n{e}[/bold red]")

    async def rag_and_get_context(self, user_query: str, user_id: str, conversation_id: str):
        """
        根据用户当前问题，搜索历史聊天信息
        """

        # 1. 在向量数据库中进行RAG搜索
        rprint("Step 1: 正在进行向量搜索...")
        # asearch 返回的是 Document 对象
        retrieved_docs = await self.vector_store.asimilarity_search(user_query)
        rprint(retrieved_docs)

        if not retrieved_docs:
            rprint("向量搜索未找到相关文档。")
            return []

        # 2. 从检索到的文档元数据中提取 message_id 列表
        rprint("Step 2: 从元数据中提取 message_ids...")
        target_message_ids = []
        for doc in retrieved_docs:
            if "message_ids" in doc.metadata:
                target_message_ids.extend(doc.metadata["message_ids"])

        # 去重
        unique_message_ids = list(set(target_message_ids))
        rprint(f"找到相关的 message_ids: {unique_message_ids}")

        if not unique_message_ids:
            return []

        # 3. 使用新的方法，从Postgres中精确、高效地获取这些消息
        rprint("Step 3: 从Postgres中精确获取消息...")
        # 注意：这里的 conversation_id 理论上可以从 doc.metadata 中获取，
        # 这样可以跨会话检索，但这里为简化，假设仍在当前会话中。
        db = self._get_db(conversation_id)

        retrieved_messages = await db.aget_messages_by_ids(unique_message_ids)

        rprint("成功获取到的历史消息上下文:")
        # rprint(retrieved_messages)

        return retrieved_messages

    async def _call_model(self, state: AgentState, config: RunnableConfig):
        """
        调用LLM
        """
        start_time = datetime.now()
        print(f"开始处理请求 (Thread ID: {config['configurable']['thread_id']})")

        current_round_messages = self._filter_messages(state)
        last_human_message = None
        memories = []
        histories = []
        for message in current_round_messages:
            if isinstance(message, HumanMessage):
                last_human_message = message
                break
        if last_human_message is not None:
            # memories = await self.store_memory_manager.asearch(
            #     query=last_human_message.content,
            #     config=config
            # )
            user_id = config["configurable"]["user_id"]
            conversation_id = config["configurable"]["conversation_id"]
            histories = await self.rag_and_get_context(
                user_query=last_human_message.content,
                user_id=user_id,
                conversation_id=conversation_id
            )
        rprint(
            f"全部消息数量: {len(state["messages"])}, 本轮消息数量: {len(current_round_messages)}\n当前记忆: {memories}")

        system_message = SystemMessagePromptTemplate.from_template(
            system_prompt
        ).format(memories=memories, histories=get_conversation(histories))
        rprint(f"system_message: {system_message.content}")

        response = await self.llm.ainvoke([system_message, *current_round_messages])

        all_messages = current_round_messages + [response]

        # 异步保存记忆
        asyncio.create_task(self._save_memory(all_messages, config))

        total_time = datetime.now() - start_time
        print(
            f"完成处理请求 (Thread ID: {config['configurable']['thread_id']}), 花费时间: {total_time}")

        return {"messages": [response]}

    async def arun(self, *, user_input: str, user_id: str, conversation_id: str) -> AIMessage:
        """
        这是暴露给外部调用的主方法。
        它封装了配置创建和调用 agent 的逻辑。
        """
        config = self._create_config(user_id, conversation_id)

        response = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        return response["messages"][-1]

    async def aget_state(self, user_id: str, conversation_id: str):
        config = self._create_config(user_id, conversation_id)
        try:
            current_state = await self.agent.aget_state(config=config)
            return [{"content": message.content, "type": message.type, "id": message.id} for message in current_state.values.get("messages", [])]
        except Exception as e:
            print(f"获取状态出错: {e}")
            return []

    async def aget_store(self, user_id: str):
        try:
            current_store = await self.agent.store.asearch(
                ("memories", user_id)
            )
            rprint(current_store)
            return [{"namespace": item.namespace, "key": item.key, "value": item.value} for item in current_store]
        except Exception as e:
            print(f"获取store出错: {e}")
            return []
