from fastapi import FastAPI
from fastapi import APIRouter, Request
from fastapi_utils.cbv import cbv


hello_webservice_api_router = APIRouter()

@cbv(hello_webservice_api_router)
class HelloWorldWebService:    
    @hello_webservice_api_router.get("/hello")
    async def read_root(self, request: Request):
        return {"Hello": "Hello World"}
    
