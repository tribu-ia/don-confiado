from fastapi import FastAPI
from fastapi import APIRouter, Request
from fastapi_utils.cbv import cbv


hello_webservice_api_router = APIRouter()

#@cbv es un decorador que le dice a FastAPI: "Voy a definir un grupo de endpoints (rutas) dentro de la clase que viene a continuación".
#hello_webservice_api_router es el objeto APIRouter al que se asociarán todas las rutas definidas dentro de esta clase. 
#En resumen, este decorador conecta la clase HelloWorldWebService con el router.
#
@cbv(hello_webservice_api_router)

# A standard Python class that groups one or more related API endpoints.
class HelloWorldWebService:
    #This decorator registers the read_root method to handle HTTP GET requests that are sent to the /hello path.
    @hello_webservice_api_router.get("/hello")
    
    # async def read_root(self, request: Request)::
    # This is the asynchronous function that executes when a request hits the /hello endpoint.
    # async def makes the function non-blocking, which is efficient for web servers.
    # It takes a Request object as an argument, which contains details about the incoming HTTP request.
    
    async def read_root(self, request: Request):
        #The function returns a Python dictionary. The web framework (like FastAPI) automatically converts this dictionary into a JSON response that is sent back to the client.
        return {"Hello": "Hello World"}
    
