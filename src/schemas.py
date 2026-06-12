from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str

class Request(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None