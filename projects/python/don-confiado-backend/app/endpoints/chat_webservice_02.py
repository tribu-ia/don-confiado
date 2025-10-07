from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ai.schemas.facturas import FacturaColombiana, UserIntention   
    
"""Chat endpoints sin utilizar helpers de memoria de LangChain.

Se usa un almacenamiento en memoria simple (dict + listas) por usuario
para construir el contexto de conversación y se generan prompts como cadenas.
"""

from endpoints.dto.message_dto import (ChatRequestDTO)
from supabase import create_client, Client
from business.dao.producto_dao import ProductoDAO
from business.dao.tercero_dao import TerceroDAO
from business.entities.producto import Producto
from business.entities.tercero import Tercero
from business.common.connection import SessionLocal
import uuid
from datetime import datetime


donconfiado_system_prompt = """ROLE:
                Don Confiado, un asistente de inteligencia artificial que actúa como un asesor
                empresarial confiable, experimentado y cercano. Es el socio virtual de las
                empresas que buscan organización, claridad y crecimiento.

                TASK:
                Mantener una conversación amigable con el usuario, siempre iniciando con un saludo
                personalizado y preguntando su nombre. Después del saludo inicial, presentarse
                brevemente como Don Confiado en 1–2 frases, explicando en qué consiste sin entrar
                en demasiados detalles. Luego, responder de manera clara y concisa cualquier
                pregunta usando solo la información provista en el contexto.

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
                - Se convierte en un “socio virtual” que siempre está disponible.

                Estilo de comunicación:
                - Amigable, cercano y claro, como un asesor de confianza.
                - Sin jerga técnica ni financiera innecesaria.
                - Siempre ofrece tranquilidad + acción: diagnóstico + recomendación.

                CONSTRAINTS:
                - Nunca inventar datos financieros concretos (montos, fechas, cifras).
                - No inventar capacidades o información que no esté en este contexto.
                - Mantener siempre un tono seguro, confiable y humano.
                - Hablar en primera persona como “Don Confiado”.

                OUTPUT_POLICY:
                - Responde en 2–4 frases como máximo.
                - Siempre comienza saludando y pidiendo el nombre del usuario.
                - Después del saludo, preséntate brevemente (1–2 frases).
                - Luego responde a la pregunta del usuario con la información disponible.
                - Si no sabes algo, dilo claramente en lugar de inventar.

                INSTRUCCIONES ADICIONALES:
                - Siempre empieza con un saludo y la pregunta por el nombre del usuario.
                - Mantén todas las respuestas cortas, claras y útiles.
                - Sé amigable y profesional en cada respuesta.
                """


# --- Router y clase del servicio de chat ---
chat_webservice_api_router_02 = APIRouter()



@cbv(chat_webservice_api_router_02)
class ChatWebService02:
    

    def __init__(self):
        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self._conversations = {}

    #Busca una conversacion y si no existe la crea
    def find_conversation(self, conversation_id: str):
        if conversation_id in self._conversations.keys():
            return self._conversations[conversation_id]
        else:
            conversation = []
            conversation.append(SystemMessage(content=donconfiado_system_prompt))
            self._conversations[conversation_id] = conversation
            return conversation 
        

    def _history_as_text(self,user_id: str) -> str:
        lines = []
        conversation = self.find_conversation(user_id)
        for msg in conversation:
            if isinstance(msg, HumanMessage):
                lines.append(f"Usuario: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Asistente: {msg.content}")
        #    elif isinstance(msg, SystemMessage):
        # opcional, si quieres incluirlo
        #        lines.append(f"Sistema: {msg.content}")
        return "\n".join(lines)
    
    def _save_product(self, payload):
        """Save a product to the database using the extracted payload."""
        session = SessionLocal()
        try:
            producto_dao = ProductoDAO(session)
            tercero_dao = TerceroDAO(session)
            
            # Generate a unique SKU if not provided
            if payload.sku:
                sku = payload.sku
            else:
                sku_base = payload.nombre.replace(" ", "_").upper()[:20]
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                sku = f"{sku_base}_{timestamp}"
            
            # Find or lookup provider if specified
            proveedor_id = None
            if payload.proveedor:
                # Try to find provider by numero_documento (NIT) or razon_social
                proveedor = tercero_dao.findByNumeroDocumento(payload.proveedor)
                if proveedor:
                    proveedor_id = proveedor.id
                    print(f"📦 Provider found: {proveedor.razon_social or proveedor.nombres}")
                else:
                    print(f"⚠️ Provider '{payload.proveedor}' not found in database")
            
            # Create the minimal product entity
            nuevo_producto = Producto(
                sku=sku,
                nombre=payload.nombre,
                precio_venta=payload.precio_venta,
                cantidad=payload.cantidad,
                proveedor_id=proveedor_id
            )
            
            # Save the product (create method already commits)
            saved_product = producto_dao.create(nuevo_producto)
            
            print(f"✅ Product saved successfully: {saved_product}")
            return saved_product
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error saving product: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al guardar el producto: {str(e)}")
        finally:
            session.close()
    
    def _save_tercero(self, payload, tipo_tercero: str):
        """Save a tercero (provider or client) to the database using the extracted payload.
        
        Args:
            payload: PayloadCreateProvider or PayloadCreateClient with the extracted data
            tipo_tercero: Either 'proveedor' or 'cliente'
        """
        session = SessionLocal()
        try:
            tercero_dao = TerceroDAO(session)
            
            # Determine tipo_documento based on NIT format (simple heuristic)
            # If numeric and length suggests company NIT, use 'NIT', otherwise 'CC'
            tipo_documento = 'NIT'
            
            # Map payload fields to Tercero entity
            # For providers and clients, we'll use razon_social for business names
            nuevo_tercero = Tercero(
                tipo_documento=tipo_documento,
                numero_documento=payload.nit,
                razon_social=payload.nombre,
                telefono_celular=getattr(payload, 'telefono', None),
                tipo_tercero=tipo_tercero,
                direccion=getattr(payload, 'direccion', None)
            )
            
            # Save the tercero (create method already commits)
            saved_tercero = tercero_dao.create(nuevo_tercero)
            
            print(f"✅ {tipo_tercero.capitalize()} saved successfully: {saved_tercero}")
            return saved_tercero
            
        except Exception as e:
            session.rollback()
            print(f"❌ Error saving {tipo_tercero}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al guardar el {tipo_tercero}: {str(e)}")
        finally:
            session.close()
    
    # --- v1.1: Clasificación de intención + extracción y registro de distribuidor ---
    @chat_webservice_api_router_02.post("/api/chat_v2.0")
    async def chat_with_structure_output(self, request: ChatRequestDTO):
        global GOOGLE_API_KEY
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai",  api_key=self.GOOGLE_API_KEY)


        conversation = self.find_conversation(request.user_id)

        # Registrar el mensaje actual en memoria y construir historial
        user_input = request.message
        conversation.append(HumanMessage(content=user_input))

        
        history_text = self._history_as_text(request.user_id)

        # Usar el modelo Pydantic para clasificación de intención
        model_with_structure = llm.with_structured_output(UserIntention)

        # Clasificación de intención basada en el prompt del Colab
        classify_text = (
            "Eres un asistente para gestión comercial.\n"
            "Clasifica la intención del usuario y extrae los datos mencionados según el schema.\n"
            "Intenciones disponibles: create_provider, create_client, create_product, other, none, bye.\n"
            "- 'create_provider': cuando el usuario quiere crear un proveedor\n"
            "- 'create_client': cuando el usuario quiere crear un cliente\n"
            "- 'create_product': cuando el usuario quiere crear un producto\n"
            "- 'other': conversación casual u otro propósito\n"
            "- 'none': sin intención clara\n"
            "- 'bye': despedida\n\n"
            f"Historial:\n{history_text}\n\n"
            f"Último mensaje del usuario: {user_input}"
        )

        result = model_with_structure.invoke(classify_text)
        
        # Imprimir resultado de detección de intención
        print("=============== INTENTION DETECTION RESULT ===============")
        print(f"User Intention: {result.userintention}")
        print(f"Payload Provider: {result.payload_provider}")
        print(f"Payload Client: {result.payload_client}")
        print(f"Payload Product: {result.payload_product}")
        print(f"Full Result: {result}")
        print("=========================================================")

        # Handle create_product intention - save the product
        saved_product = None
        product_saved_successfully = False
        if result.userintention == "create_product" and result.payload_product:
            try:
                saved_product = self._save_product(result.payload_product)
                product_saved_successfully = True
                print(f"🎉 Product '{saved_product.nombre}' saved with SKU: {saved_product.sku}")
            except Exception as e:
                print(f"⚠️ Failed to save product: {str(e)}")
                # Don't raise the exception, just log it and continue with the conversation

        # Handle create_provider intention - save the provider
        saved_provider = None
        provider_saved_successfully = False
        if result.userintention == "create_provider" and result.payload_provider:
            try:
                saved_provider = self._save_tercero(result.payload_provider, 'proveedor')
                provider_saved_successfully = True
                print(f"🎉 Provider '{saved_provider.razon_social}' saved with ID: {saved_provider.id}")
            except Exception as e:
                print(f"⚠️ Failed to save provider: {str(e)}")
                # Don't raise the exception, just log it and continue with the conversation

        # Handle create_client intention - save the client
        saved_client = None
        client_saved_successfully = False
        if result.userintention == "create_client" and result.payload_client:
            try:
                saved_client = self._save_tercero(result.payload_client, 'cliente')
                client_saved_successfully = True
                print(f"🎉 Client '{saved_client.razon_social}' saved with ID: {saved_client.id}")
            except Exception as e:
                print(f"⚠️ Failed to save client: {str(e)}")
                # Don't raise the exception, just log it and continue with the conversation

        # Generate a simple response
        ai_result = llm.invoke(conversation)
        reply = getattr(ai_result, "content", str(ai_result))
        conversation.append(AIMessage(content=reply))
        print("===REPLY===")
        print(reply)
        return {
            "userintention": result.userintention,
            "reply": reply,
            "payload_provider": result.payload_provider.model_dump() if result.payload_provider else None,
            "payload_client": result.payload_client.model_dump() if result.payload_client else None,
            "payload_product": result.payload_product.model_dump() if result.payload_product else None,
            "product_saved": product_saved_successfully,
            "saved_product_id": saved_product.id if saved_product else None,
            "saved_product_sku": saved_product.sku if saved_product else None,
            "provider_saved": provider_saved_successfully,
            "saved_provider_id": saved_provider.id if saved_provider else None,
            "saved_provider_name": saved_provider.razon_social if saved_provider else None,
            "client_saved": client_saved_successfully,
            "saved_client_id": saved_client.id if saved_client else None,
            "saved_client_name": saved_client.razon_social if saved_client else None,
        }

        