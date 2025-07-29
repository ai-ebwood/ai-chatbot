# main.py
from rich import print as rprint
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.vectorstores.base import VectorStore

# 导入我们的服务和 LangGraph/LangChain 的组件
from agent_service import AgentService
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langchain_postgres import PGVectorStore, PGEngine
from langchain_openai import OpenAIEmbeddings

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用的生命周期管理器。
    在应用启动时，它会运行 yield 之前的所有代码。
    在应用关闭时，它会运行 yield 之后的所有代码。
    """
    print("Application startup: Initializing resources...")
    pg_conn = os.environ["NEON_DB_URL"]
    async_pg_conn = os.environ["NEON_DB_URL_ASYNC"]

    TABLE_NAME = "ai_chatbot_conversation"

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    # 使用 with 语句来正确管理数据库连接的生命周期
    async with (
        AsyncPostgresSaver.from_conn_string(pg_conn) as checkpointer,
        AsyncPostgresStore.from_conn_string(
            pg_conn,
            index={"dims": 1536, "embed": embeddings}
        ) as store
    ):
        engine = create_async_engine(async_pg_conn)

        pg_engine = PGEngine.from_engine(engine=engine)

        try:
            await pg_engine.ainit_vectorstore_table(
                table_name=TABLE_NAME,
                vector_size=1536,
            )
        except Exception as e:
            print(f"创建表失败: {e}")

        pg_vector = await PGVectorStore.create(
            engine=pg_engine,
            embedding_service=embeddings,
            table_name=TABLE_NAME
        )
        # 新数据库需要setup
        # await checkpointer.setup()
        # await store.setup()
        # 创建 AgentService 实例，注入依赖项
        agent_service = AgentService(
            checkpointer=checkpointer, store=store, async_pg_conn=async_pg_conn, vector_store=pg_vector)

        # 将实例附加到 app.state，以便在请求处理程序中访问
        app.state.agent_service = agent_service

        print("Resources initialized and agent is ready.")
        yield  # 应用在此处运行

        # yield 之后的部分在应用关闭时执行
        # 'with' 语句会自动处理 checkpointer.close() 和 store.close()
        print("Application shutdown: Resources have been released.")

# --- FastAPI 应用设置 ---
app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:5173"  # 你的前端地址
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatParams(BaseModel):
    question: str
    user_id: str

# --- API 端点 ---


@app.post("/chat")
async def ask_question(params: ChatParams, request: Request):
    """
    处理聊天请求。
    通过 request.app.state 访问共享的 agent_service 实例。
    """
    # 从 app.state 获取在启动时创建的服务实例
    agent_service: AgentService = request.app.state.agent_service

    # 调用服务的异步方法
    result = await agent_service.arun(user_input=params.question, user_id=params.user_id)

    return {"content": result.content}


@app.get("/")
def read_root():
    return {"status": "Agent server is running"}


@app.get("/state")
async def get_state(user_id: str, request: Request):
    # 从 app.state 获取在启动时创建的服务实例
    agent_service: AgentService = request.app.state.agent_service

    # 调用服务的异步方法
    result = await agent_service.aget_state(user_id)

    return result


@app.get("/store")
async def get_store(user_id: str, request: Request):
    # 从 app.state 获取在启动时创建的服务实例
    agent_service: AgentService = request.app.state.agent_service

    # 调用服务的异步方法
    result = await agent_service.aget_store(user_id)

    return result
