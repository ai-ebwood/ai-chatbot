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
    BaseMessage,
    RemoveMessage
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

    def _filter_messages_with_round(self, state: AgentState, round: int = 1):
        """
        从消息列表出过滤出最后{round}轮次HumanMessage开头的消息.

        Args:
            state: 状态，包含消息列表
            round: 过滤轮次, 默认为1次，也就是本轮

        Return: 
            keep_messages: 保留的消息
            remove_messages: 要删除的消息
        """
        all_messages = state["messages"]
        keep_messages = []
        remove_messages = []
        for message in reversed(all_messages):
            keep_messages.append(message)
            if not isinstance(message, HumanMessage):
                continue
            round -= 1
            if round <= 0:
                break
        if round <= 0:
            keep_count = len(keep_messages)
            remove_messages = [RemoveMessage(message.id)
                               for message in all_messages[:-keep_count]]
        return keep_messages, remove_messages

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
        retrieved_docs = await self.vector_store.asimilarity_search_with_score(user_query, k=2)
        rprint(retrieved_docs)
        rprint("\n")

        if not retrieved_docs:
            rprint("向量搜索未找到相关文档。")
            return []

        # 2. 从检索到的文档元数据中提取 message_id 列表
        rprint("Step 2: 从元数据中提取 message_ids...")
        target_message_ids = []
        for (doc, score) in retrieved_docs:
            if "message_ids" in doc.metadata:
                target_message_ids.extend(doc.metadata["message_ids"])

        # 去重
        unique_message_ids = list(set(target_message_ids))

        if not unique_message_ids:
            return []

        # 3. 使用新的方法，从Postgres中精确、高效地获取这些消息
        rprint("Step 3: 从Postgres中精确获取消息...")
        # 注意：这里的 conversation_id 理论上可以从 doc.metadata 中获取，
        # 这样可以跨会话检索，但这里为简化，假设仍在当前会话中。
        db = self._get_db(conversation_id)

        retrieved_messages = await db.aget_messages_by_ids(unique_message_ids)

        rprint("成功获取到的历史消息上下文:\n")
        # rprint(retrieved_messages)

        return retrieved_messages

    async def _call_model(self, state: AgentState, config: RunnableConfig):
        """
        调用LLM
        """
        start_time = datetime.now()
        rprint(f"开始处理请求 (Thread ID: {config['configurable']['thread_id']})\n")

        current_round_messages, remove_messages = self._filter_messages_with_round(
            state, round=2)
        last_human_message = None
        memories = []
        histories = []
        for message in current_round_messages:
            if isinstance(message, HumanMessage):
                last_human_message = message
                break
        if last_human_message is not None:
            memories = await self.store_memory_manager.asearch(
                query=last_human_message.content,
                config=config
            )
            user_id = config["configurable"]["user_id"]
            conversation_id = config["configurable"]["conversation_id"]
            histories = await self.rag_and_get_context(
                user_query=last_human_message.content,
                user_id=user_id,
                conversation_id=conversation_id
            )
        rprint(
            f"全部消息数量: {len(state["messages"])}, 本轮消息数量: {len(current_round_messages)}\n当前记忆: {memories}\n")

        system_message = SystemMessagePromptTemplate.from_template(
            system_prompt
        ).format(memories=memories, histories=get_conversation(histories))
        rprint(f"system_message: {system_message.content}\n")

        response = await self.llm.ainvoke([system_message, *current_round_messages])

        all_messages = current_round_messages + [response]

        # 异步保存记忆
        asyncio.create_task(self._save_memory(all_messages, config))

        total_time = datetime.now() - start_time
        rprint(
            f"完成处理请求 (Thread ID: {config['configurable']['thread_id']}), 花费时间: {total_time}\n")

        return {"messages": remove_messages + [response]}

    async def arun(self, *, user_input: str, user_id: str, conversation_id: str) -> AIMessage:
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
            rprint(f"{current_store}\n")
            return [{"namespace": item.namespace, "key": item.key, "value": item.value} for item in current_store]
        except Exception as e:
            rprint(f"获取store出错: {e}\n")
            return []

    async def aget_conversation(self, user_id: str, conversation_id: str):
        try:
            db = self._get_db(conversation_id)
            messages = await db.aget_messages()
            return messages

        except Exception as e:
            rprint(f"获取conversation出错: {e}")
            return []
