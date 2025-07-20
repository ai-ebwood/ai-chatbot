import os
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables.config import RunnableConfig


class Configuration(BaseModel):
    model: str = Field(
        default="gpt-4o-mini"
    )
    model_provider: str = Field(
        default="openai"
    )
    max_token_limit: int = Field(
        default=10*1000
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None):
        """Create a Configuration from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        print(field_names)
        values: dict[str, Any] = {
            field_name: os.environ.get(
                field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
