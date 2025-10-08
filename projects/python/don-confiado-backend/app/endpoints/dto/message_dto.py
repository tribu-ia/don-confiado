from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class MessageDTO(BaseModel):
    message: str
    source: str
    destination: str


class ChatRequestDTO(BaseModel):
    message: str
    user_id: str
    mime_type: Optional[str] = None
    file_base64: Optional[str] = None 
    def __str__(self):
        return f"ChatRequestDTO(message={self.message}, user_id={self.user_id}, mime_type={self.mime_type}, file_base64={self.file_base64})"
    