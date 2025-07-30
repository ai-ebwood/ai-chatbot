from pydantic import BaseModel
from langgraph.graph import MessagesState

from pydantic import Field
from typing import Annotated
import operator


class State(MessagesState):
    """Agent state"""
    exceed_tokens: bool = Field(
        default=False
    )
    total_tokens: Annotated[int, operator.add] = Field(
        default=0
    )


class AgentState(MessagesState):
    pass


class UserProfile(BaseModel):
    name: str | None = None
    age: int | None = None
    user_experiences: list[str] = []
    preferences: dict | None = None
