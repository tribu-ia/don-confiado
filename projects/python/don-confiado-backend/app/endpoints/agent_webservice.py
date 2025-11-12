# Standard library imports
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv

from .dto.message_dto import ChatRequestDTO
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware 

#from langchain.agents import create_tool_calling_agent, AgentExecutor
from ai.agents.chatbot_agent.chatbot_agent import create_tools_array
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command




# =============================================================================
# CONSTANTS
# =============================================================================

DONCONFIADO_SYSTEM_PROMPT = """ROLE:
Don Confiado, un asistente de inteligencia artificial que actúa como un asesor
empresarial confiable, experimentado y cercano. Es el socio virtual de las
empresas que buscan organización, claridad y crecimiento.

TASK:
Mantener una conversación amigable y contextual con el usuario. Si es la primera
interacción, saluda y preséntate brevemente. En conversaciones posteriores, responde
directamente a lo que el usuario está preguntando o pidiendo, considerando el contexto
de la conversación.

CONTEXT:
Don Confiado está diseñado para pequeñas y medianas empresas (PYMES) y emprendedores
que desean enfocarse en vender y crecer, sin descuidar la administración. Su misión
es quitar la carga administrativa que suele consumir tiempo y energía, para que los
empresarios puedan enfocarse en lo más importante: la estrategia y los clientes.

Capacidades principales:
1. Flujo de caja:
- Monitorear ingresos y egresos.
- Detectar problemas de liquidez.
- Recomendar acciones concretas para mantener estabilidad financiera.
2. Inventario:
- Organizar productos y niveles de stock.
- Generar alertas cuando un producto esté por agotarse.
- Predecir necesidades de reabastecimiento con base en ventas pasadas.
3. Proveedores y distribuidores:
- Registrar y organizar proveedores confiables.
- Recordar pagos y fechas clave.
- Optimizar la logística para reducir costos y tiempos de entrega.
4. Ventas con IA:
- Detectar patrones de compra en clientes.
- Recomendar promociones o estrategias personalizadas.
- Identificar productos de alto rendimiento y oportunidades de mercado.

Clientes objetivo:
- Emprendedores que manejan todo solos y necesitan organización.
- PYMES que buscan crecer sin contratar un gran equipo administrativo.
- Negocios en expansión que quieren controlar caja, stock y proveedores.

Propuesta de valor:
- Ahorra tiempo al automatizar tareas administrativas.
- Genera confianza con reportes y recomendaciones claras.
- Ayuda a vender más gracias a la inteligencia de datos.
- Se convierte en un "socio virtual" que siempre está disponible.

Estilo de comunicación:
- Amigable, cercano y claro, como un asesor de confianza.
- Sin jerga técnica ni financiera innecesaria.
- Siempre ofrece tranquilidad + acción: diagnóstico + recomendación.

CONSTRAINTS:
- Nunca inventar datos financieros concretos (montos, fechas, cifras).
- No inventar capacidades o información que no esté en este contexto.
- Mantener siempre un tono seguro, confiable y humano.
- Hablar en primera persona como "Don Confiado".

OUTPUT_POLICY:
- Responde en 2–4 frases como máximo.
- Si es la primera interacción: saluda, preséntate brevemente y pregunta el nombre.
- En conversaciones posteriores: responde directamente a lo que el usuario pregunta.
- Considera el contexto de la conversación anterior.
- Si no sabes algo, dilo claramente en lugar de inventar.

INSTRUCCIONES ADICIONALES:
- Lee y responde al mensaje específico del usuario, no uses respuestas genéricas.
- Mantén todas las respuestas cortas, claras y útiles.
- Sé amigable y profesional en cada respuesta.
- Adapta tu respuesta al contexto de la conversación.
"""


agent_webservice_api_router = APIRouter()
@cbv(agent_webservice_api_router)
class AgentWebService:
    """
    Chat service for Don Confiado AI assistant.
    
    Handles multimodal conversations (text, audio, images) with intention detection
    and automatic data extraction from invoices and user inputs.
    """

    _conversations = {}
    _pending_approval = {}
    _agent = None
    _inMemorySaver = InMemorySaver()
    
    def __init__(self):
        """Initialize the chat service with environment variables and conversation storage."""
        print("*******************INITIALIZING CHAT WEB SERVICE**************")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.gemini_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai",  api_key=self.GOOGLE_API_KEY)

        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=self.GOOGLE_API_KEY)
        tools = create_tools_array()
        middleware=[
        HumanInTheLoopMiddleware( 
            interrupt_on={
                "buscar_terceros_tool": True,  # All decisions (approve, edit, reject) allowed
                "buscar_por_rango_de_precio": {"allowed_decisions": ["approve", "reject"]},  # No editing allowed
            },
            description_prefix="Tool execution pending approval",
        ),
        ]
        self._agent = create_agent(model = llm,tools = tools,middleware = middleware, checkpointer = self._inMemorySaver)


    # =============================================================================
    # CONVERSATION MANAGEMENT UTILITIES
    # =============================================================================
    
    def find_conversation(self, conversation_id: str):
        """
        Find an existing conversation or create a new one.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Returns:
            List of messages for the conversation
        """

        global DONCONFIADO_SYSTEM_PROMPT
        print("LOOKING FOR CONVERSATION ID:", conversation_id)
        for key in self._conversations.keys():
            print("EXISTING CONVERSATION KEY:", key)
        if conversation_id in self._conversations.keys():
            return self._conversations[conversation_id]
        else:
            conversation = []
            conversation.append(SystemMessage(content=DONCONFIADO_SYSTEM_PROMPT))            
            self._conversations[conversation_id] = conversation
            return conversation 

    
    # =============================================================================
    # MAIN CHAT ENDPOINT
    # =============================================================================
    
    @agent_webservice_api_router.post("/api/chat_v3.0")
    async def process_incomming_message(self, request: ChatRequestDTO):
        """
        Main chat endpoint with intention detection and multimodal support.
        
        Handles text, audio, and image inputs with automatic data extraction
        and entity creation (products, providers, clients).
        
        Args:
            request: ChatRequestDTO with user message and optional file data
            
        Returns:
            Dict with chat response, detected intention, and saved entities
        """
        
        print("=========REQUEST=========")
        print(request)
        print("=========================")


        
        # Get or create conversation
        print("FINDING CONVERSATION FOR USER ID:", request.user_id)
        conversation = self.find_conversation(request.user_id)
        config = {"configurable": {"thread_id": request.user_id}} 
        print("Configurable", config)

        print("=========CONVERSATION=========")
        for c in conversation:
            print(c)
        print("=============================")

        if request.user_id in self._pending_approval:
            print("========= PENDING APPROVAL DETECTED =========")
            self._pending_approval.pop(request.user_id)
            response = None
            if  request.message.lower() in ["yes", "approve", "aprove", "si", "sí", "y"]:
                response = self._agent.invoke(Command( resume={"decisions": [{"type": "approve"}] } ), config=config)
            else:
                response = self._agent.invoke(Command( resume={"decisions": [{"type": "reject"}]} ), config=config)

            print("========= APPROVAL RESPONSE=========")
            print(type(response))
            response_dto = {"answer": response["messages"][-1].content}
            conversation.append(response["messages"][-1])          
            return response_dto

        
        conversation.append(HumanMessage(content=request.message))        
        
        response = self._agent.invoke({"messages": conversation} ,config=config , verbose=True)
        
        if "__interrupt__" in response:  #Manejo de interrupciones para aprobaciones humanas
            print("========= INTERRUPT DETECTED =========")
            print(response["__interrupt__"])
            interrupt_message = """Lo que quieres hacer requiere tu aprobación. Responde si o yes para aprobar, o no o reject para rechazar."""
            response_dto = {"answer":  interrupt_message }
            self._pending_approval[request.user_id] = response["__interrupt__"][0]
            return response_dto 

        print("========= FUNCIONA  RESPONSE=========")
        conversation.append(response["messages"][-1])
        print(type(response))
        print(response)
        response_dto = {"answer": response["messages"][-1].content}
        print("=========RESPONSE=========")
        print(response_dto)

        return response_dto
