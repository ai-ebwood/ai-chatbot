from datetime import datetime
import asyncio
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()


async def run_agent(*, model: str | None = None, model_provider: str | None = None, question: str) -> str:
    llm = init_chat_model(
        model=model or "gpt-4o-mini",
        model_provider=model_provider or "openai"
    )
    result = await llm.ainvoke(question)
    return result.model_dump()


async def main():
    print(f"开始: {datetime.now()}")
    result = await run_agent(question="又下雨啦")
    print(result)
    print(f"结束: {datetime.now()}")

if __name__ == "__main__":
    asyncio.run(main())
