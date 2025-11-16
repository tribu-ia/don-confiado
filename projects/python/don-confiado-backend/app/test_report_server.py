from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from endpoints.hello_world_webservice import hello_webservice_api_router
from endpoints.report_webservice import report_webservice_api_router
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


if __name__ == "__main__":
    app = FastAPI()
    app.include_router(hello_webservice_api_router)
    app.include_router(report_webservice_api_router)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)

