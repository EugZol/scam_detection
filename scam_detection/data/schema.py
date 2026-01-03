from pydantic import BaseModel


class MessageData(BaseModel):
    text: str
    label: str


class MessageBatch(BaseModel):
    texts: list[str]
    labels: list[str]
