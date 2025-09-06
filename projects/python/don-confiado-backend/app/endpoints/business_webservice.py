from fastapi import FastAPI
from fastapi import APIRouter, Request
from fastapi_utils.cbv import cbv
from pydantic import BaseModel
from endpoints.dto.message_dto import MessageDTO



business_webservice_api_router = APIRouter()

@cbv(business_webservice_api_router)
class HelloWorldWebService:    
    @business_webservice_api_router.post("/api/business/message")
    async def process_message(self, message_dto: MessageDTO):
        # Procesar el mensaje recibido
        response = {
            "status": "success",
            "received_message": message_dto.message,
            "from": message_dto.source,
            "to": message_dto.destination,
            "processed_at": "2025-09-05T12:00:00Z"  # Puedes usar datetime.utcnow()
        }
        
        return response