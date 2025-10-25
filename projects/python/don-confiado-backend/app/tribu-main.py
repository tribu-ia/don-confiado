from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from endpoints.hello_world_webservice import HelloWorldWebService, hello_webservice_api_router
from endpoints.business_webservice import business_webservice_api_router
from endpoints.chat_webservice_02 import chat_webservice_api_router_02
from endpoints.chat_clase_03 import chat_clase_03_api_router
from endpoints.chat_clase_04 import graphrag_api_router
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


if __name__ == "__main__":
    app = FastAPI()
    app.include_router(hello_webservice_api_router)
    app.include_router(business_webservice_api_router)
    app.include_router(chat_webservice_api_router_02)
    app.include_router(chat_clase_03_api_router)
    app.include_router(graphrag_api_router)
    uvicorn.run(app, host="127.0.0.1", port=8000)
