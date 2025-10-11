from fastapi import FastAPI
import uvicorn
from endpoints.hello_world_webservice import HelloWorldWebService, hello_webservice_api_router
from endpoints.business_webservice import business_webservice_api_router
from endpoints.chat_webservice import chat_webservice_api_router

if __name__ == "__main__":
    app = FastAPI()
    app.include_router(hello_webservice_api_router)
    app.include_router(business_webservice_api_router)
    app.include_router(chat_webservice_api_router)
    uvicorn.run(app, host="127.0.0.1", port=8000)
