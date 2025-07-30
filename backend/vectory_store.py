
from langchain_postgres import PGEngine, PGVectorStore
from psycopg import AsyncConnection
from sqlalchemy.ext.asyncio import create_async_engine

from constants import CONVERSATION_VECTOR_TABLE_NAME
from rich import print as rprint
from langchain_core.embeddings import Embeddings


async def create_pg_vector(async_pg_conn: str, embeddings: Embeddings):
    engine = create_async_engine(async_pg_conn)
    pg_engine = PGEngine.from_engine(engine=engine)
    try:
        # 数据库第一次创建需要，后续可以注释掉
        await pg_engine.ainit_vectorstore_table(
            table_name=CONVERSATION_VECTOR_TABLE_NAME,
            vector_size=1536,
        )
    except Exception as e:
        rprint(f"[bold red]创建表失败: {e}[/bold red]")
    pg_vector = await PGVectorStore.create(
        engine=pg_engine,
        embedding_service=embeddings,
        table_name=CONVERSATION_VECTOR_TABLE_NAME
    )
    return pg_vector

async def create_qdrant_vector(embeddings: Embeddings):
    pass