from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn
from endpoints.hello_world_webservice import HelloWorldWebService, hello_webservice_api_router
from endpoints.business_webservice import business_webservice_api_router
from endpoints.chat_webservice_02 import chat_webservice_api_router_02
from endpoints.chat_clase_03 import chat_clase_03_api_router
from endpoints.agent_webservice import agent_webservice_api_router
from endpoints.chat_clase_04 import graphrag_api_router
from endpoints.report_webservice import report_webservice_api_router
import os
import logging
from pathlib import Path

load_dotenv()

# Setup logging to file in repo
log_dir = Path(__file__).parent
log_file = log_dir / "server.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


if __name__ == "__main__":
    logger.info("Starting server...")
    logger.info(f"Log file: {log_file}")
    app = FastAPI()
    app.include_router(hello_webservice_api_router)
    app.include_router(agent_webservice_api_router)
    app.include_router(report_webservice_api_router)
    
    logger.info("Server starting on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_config=None)
