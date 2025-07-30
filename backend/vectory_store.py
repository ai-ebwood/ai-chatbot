
from qdrant_client import QdrantClient
import os
from langchain_postgres import PGEngine, PGVectorStore
from psycopg import AsyncConnection
from sqlalchemy.ext.asyncio import create_async_engine

from constants import CONVERSATION_VECTOR_TABLE_NAME
from rich import print as rprint
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import VectorParams, Distance


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
    client = QdrantClient(url=os.environ["QDRANT_URL"])

    if not client.collection_exists(CONVERSATION_VECTOR_TABLE_NAME):
        client.create_collection(
            collection_name=CONVERSATION_VECTOR_TABLE_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CONVERSATION_VECTOR_TABLE_NAME,
        embedding=embeddings,
    )
    return vector_store
