from pydantic import BaseModel
from typing import Optional


# DTO para el mensaje
class MessageDTO(BaseModel):
    message: str
    source: str
    destination: str