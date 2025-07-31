# main.py
from rich import print as rprint
import os
import psycopg
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine

# 导入我们的服务和 LangGraph/LangChain 的组件
from agent_service import AgentService
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langchain_postgres import PGVectorStore, PGEngine
from langchain_openai import OpenAIEmbeddings
from chat_history import PostgresChatMessageHistoryWithId
from vectory_store import create_pg_vector, create_qdrant_vector

from constants import (
    CONVERSATION_VECTOR_TABLE_NAME,
    CONVERSATION_TABLE_NAME
)

load_dotenv()


async def make_graph():
    print("Application startup: Initializing resources...")
    pg_conn = os.environ["NEON_DB_URL"]
    async_pg_conn = os.environ["NEON_DB_URL_ASYNC"]

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
        # await checkpointer.setup()
        # await store.setup()

        # create vector store
        # vector_store = await create_pg_vector(async_pg_conn, embeddings)
        vector_store = await create_qdrant_vector(embeddings)

        # create db
        async_connection = await psycopg.AsyncConnection.connect(pg_conn)

        # Create the table schema (only needs to be done once)
        await PostgresChatMessageHistoryWithId.acreate_tables(async_connection, CONVERSATION_TABLE_NAME)

        # 创建 AgentService 实例，注入依赖项
        agent_service = AgentService(
            checkpointer=checkpointer,
            store=store,
            vector_store=vector_store,
            db_conn_async=async_connection)
        return agent_service.agent
