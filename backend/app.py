from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # ✅ 允许的 Origin
    allow_credentials=True,
    allow_methods=["*"],             # 可按需限制为 ["GET", "POST"]
    allow_headers=["*"],             # 可按需限制
)


class ChatParams(BaseModel):
    question: str
    user_id: str

@app.post("/chat")
async def ask_question(params: ChatParams):
    result = await run_agent(question=params.question, user_id=params.user_id)
    return result
