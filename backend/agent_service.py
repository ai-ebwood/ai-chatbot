# agent_service.py

from state import (
    AgentState,
    UserProfile
)
import uuid
from langchain_core.vectorstores.base import VectorStore
from datetime import datetime
import asyncio
from dotenv import load_dotenv
from prompts import memory_prompt, generate_memory_prompt
from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import create_async_engine

from langmem import create_memory_store_manager
from rich import print as rprint
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector, PGVectorStore
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)
from langchain_core.prompts import SystemMessagePromptTemplate

# --- 配置和类型定义 (保持不变) ---
load_dotenv()


class AgentService:
    # --- AgentService 类 ---
    def __init__(self, *, checkpointer: BaseCheckpointSaver, store: BaseStore, async_pg_conn: str, vector_store: VectorStore):
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

        # 2. 初始化 langmem
        namespace_key = ("memories", "{user_id}")
        self.store_memory_manager = create_memory_store_manager(
            self.llm,
            namespace=namespace_key,
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

    def _filter_messages(self, state: AgentState):
        current_round_messages = []
        all_messages = state["messages"]
        for message in reversed(all_messages):
            current_round_messages.append(message)
            if isinstance(message, HumanMessage):
                break
        return current_round_messages

    def _get_config(self, user_id: str) -> RunnableConfig:
        return RunnableConfig(
            configurable={
                # 使用 user_id 创建一个唯一的 thread_id
                "thread_id": f"thread-for-{user_id}",
                "user_id": user_id
            }
        )

    async def _save_memory(self, messages, config):
        """
        异步保存记忆。
        修正了对 _save_conversation 的调用，正确传递 user_id。
        """
        # 从 config 中获取 user_id
        user_id = config["configurable"]["user_id"]

        # 创建两个并行的后台任务
        save_store_task = asyncio.create_task(
            self._save_store(messages, config))
        save_conversation_task = asyncio.create_task(
            self._save_conversation(messages, user_id))

        # 等待两个任务完成 (可选，但推荐用于更好的错误处理)
        await asyncio.gather(save_store_task, save_conversation_task)

    async def _save_store(self, messages, config):
        start_time = datetime.now()
        rprint(f"后台保存记忆开始: {start_time}")
        save_store_result = await self.store_memory_manager.ainvoke(
            {"messages": messages}, config=config)
        cost_time = datetime.now() - start_time
        rprint(
            f"后台保存记忆成功: {datetime.now()}, cost_time: {cost_time}\n{save_store_result}")

    async def _save_conversation(self, messages, user_id: str):
        start_time = datetime.now()
        rprint(f"后台插入会话开始: {start_time}")
        page_content = "\n".join(
            [f"{message.type}: {message.content}" for message in messages])
        # --- 核心修正 ---
        # 手动创建只包含简单数据类型的元数据，避免不可序列化的对象。
        simple_messages_metadata = []
        for msg in messages:
            simple_messages_metadata.append({
                "type": msg.type,
                "content": str(msg.content),  # 确保内容是字符串
                "id": str(msg.id) if msg.id else None
                # 不要包含 msg.response_metadata 或其他复杂对象
            })

        doc = Document(page_content=page_content, metadata={
            "messages": simple_messages_metadata,
            "user_id": user_id
        })
        doc_id = str(uuid.uuid4())
        try:
            result = await self.vector_store.aadd_documents([doc], ids=[doc_id])
            cost_time = datetime.now() - start_time
            rprint(
                f"后台插入会话成功: {datetime.now()}, cost_time: {cost_time}\nid = {doc_id}, result: \n{result}")
        except Exception as e:
            rprint(f"[bold red]后台插入会话失败: {e}[/bold red]")
            # 可以在这里进一步打印 doc 的内容以供调试
            # rprint(f"Failed to save document: {doc}")

    async def _call_model(self, state: AgentState, config: RunnableConfig):
        """
        这是 Graph 中的节点，注意它现在是类的一个方法。
        """
        start_time = datetime.now()
        print(f"开始处理请求 (Thread ID: {config['configurable']['thread_id']})")

        current_round_messages = self._filter_messages(state)
        last_human_message = None
        memories = []
        for message in current_round_messages:
            if isinstance(message, HumanMessage):
                last_human_message = message
                break
        if last_human_message is not None:
            memories = await self.store_memory_manager.asearch(
                query=last_human_message.content,
                config=config
            )
        rprint(
            f"全部消息数量: {len(state["messages"])}, 本轮消息数量: {len(current_round_messages)}\n当前记忆: {memories}")

        system_message = SystemMessagePromptTemplate.from_template(
            memory_prompt
        ).format(memories=memories)

        response = await self.llm.ainvoke([system_message, *current_round_messages])

        all_messages = current_round_messages + [response]

        # 异步保存记忆
        asyncio.create_task(self._save_memory(all_messages, config))

        total_time = datetime.now() - start_time
        print(
            f"完成处理请求 (Thread ID: {config['configurable']['thread_id']}), 花费时间: {total_time}")

        return {"messages": [response]}

    async def arun(self, *, user_id: str, user_input: str) -> AIMessage:
        """
        这是暴露给外部调用的主方法。
        它封装了配置创建和调用 agent 的逻辑。
        """
        config = self._get_config(user_id)

        response = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        return response["messages"][-1]

    async def aget_state(self, user_id: str):
        config = self._get_config(user_id)
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
