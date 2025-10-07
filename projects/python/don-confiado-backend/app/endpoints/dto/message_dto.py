from pydantic import BaseModel
from typing import Optional, List, Dict


class MessageDTO(BaseModel):
    message: str
    source: str
    destination: str


class ChatRequestDTO(BaseModel):
    message: str
    user_id: str
    mime_type: Optional[str] = None
    file_base64:Optional[str] = None 
    