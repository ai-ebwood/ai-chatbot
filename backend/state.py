from langgraph.graph import MessagesState

from pydantic import Field
from typing import Annotated, TypedDict
import operator


class State(MessagesState):
    """Agent state"""
    exceed_tokens: bool = Field(
        default=False
    )
    total_tokens: Annotated[int, operator.add] = Field(
        default=0
    )

class SummaryOutput(TypedDict):
    summary: str