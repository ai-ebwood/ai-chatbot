# main.py
from rich import print as rprint
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 导入我们的服务和 LangGraph/LangChain 的组件
from agent_service import AgentService
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
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
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    # 使用 with 语句来正确管理数据库连接的生命周期
    with (
        PostgresSaver.from_conn_string(pg_conn) as checkpointer,
        PostgresStore.from_conn_string(
            pg_conn,
            index={"dims": 1536, "embed": embeddings}
        ) as store
    ):
        # 创建 AgentService 实例，注入依赖项
        agent_service = AgentService(checkpointer=checkpointer, store=store)

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

    # Pydantic 无法直接序列化 AIMessage，所以我们返回其内容
    # return {"answer": result.content}
    return result


@app.get("/")
def read_root():
    return {"status": "Agent server is running"}
