from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str

class ChatCompRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    stream: bool = False