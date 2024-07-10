from pydantic import BaseModel, Field


class TextResponse(BaseModel):
    message: str = Field(description="Result of the operation execution")
