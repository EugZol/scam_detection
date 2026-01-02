from pydantic import BaseModel


class EmailData(BaseModel):
    text: str
    label: str


class EmailBatch(BaseModel):
    texts: list[str]
    labels: list[str]
