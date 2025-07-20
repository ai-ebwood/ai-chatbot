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
