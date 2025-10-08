from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
import os
import base64
from pathlib import Path
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
    
    def _read_file_to_base64(self, file_path: str) -> str:
        """Read a file from the given path and return it as a base64 encoded string.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Base64 encoded string of the file contents
            
        Raises:
            HTTPException if the file cannot be read
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
            
            with open(path, "rb") as file:
                file_bytes = file.read()
                return base64.b64encode(file_bytes).decode('utf-8')
                
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
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
    
    def _extract_invoice_from_image(self, llm, message_content):
        """Extract invoice data from an image using structured output.
        
        Args:
            llm: The language model instance
            message_content: List with text and image_url content
            
        Returns:
            FacturaColombiana object or None if extraction fails
        """
        try:
            # Create a model with structured output for invoice extraction
            model_with_invoice_structure = llm.with_structured_output(FacturaColombiana)
            
            # Instruction for invoice extraction (following Colab pattern)
            invoice_extraction_message = HumanMessage(content=[
                {"type": "text", "text": "Analizar la imagen y extraer los datos de la factura según el schema proporcionado. Si la imagen no contiene una factura, responde con datos vacíos o nulos."},
                message_content[1]  # The image_url content
            ])
            
            # Extract invoice data
            invoice_data = model_with_invoice_structure.invoke([invoice_extraction_message])
            
            print("=============== INVOICE EXTRACTION RESULT ===============")
            print(f"Invoice Number: {invoice_data.numeroFactura}")
            print(f"Date: {invoice_data.fechaEmision}")
            print(f"Total: {invoice_data.total}")
            print(f"Emisor: {invoice_data.emisor}")
            print(f"Items count: {len(invoice_data.items)}")
            print("=========================================================")
            
            return invoice_data
            
        except Exception as e:
            print(f"⚠️ Failed to extract invoice data: {str(e)}")
            return None
    
    def _enrich_intention_with_invoice(self, result, invoice_data):
        """Enrich the detected intention with invoice data if applicable.
        
        Simple approach: If user wants to create provider/product and we have invoice data,
        populate the payloads with invoice information.
        
        Args:
            result: UserIntention object from LLM
            invoice_data: FacturaColombiana object from image extraction
            
        Returns:
            UserIntention object with enriched payloads
        """
        if not invoice_data:
            return result
        
        from ai.schemas.facturas import PayloadCreateProvider, PayloadCreateProduct
        
        # If intention is create_provider and we have invoice data, use emisor data
        if result.userintention == "create_provider":
            print("📝 Enriching provider payload with invoice emisor data...")
            result.payload_provider = PayloadCreateProvider(
                nombre=invoice_data.emisor.razonSocial,
                nit=invoice_data.emisor.nit
            )
            print(f"✅ Provider payload enriched: {result.payload_provider.nombre}")
        
        # If intention is create_product and we have invoice items, use first item
        elif result.userintention == "create_product" and invoice_data.items:
            print("📝 Enriching product payload with invoice item data...")
            first_item = invoice_data.items[0]
            
            # Calculate price from item data
            precio = first_item.precioUnitario if first_item.precioUnitario else (
                first_item.subtotal / first_item.cantidad if first_item.subtotal and first_item.cantidad > 0 else 0
            )
            
            result.payload_product = PayloadCreateProduct(
                nombre=first_item.descripcion[:200],  # Truncate to max length
                precio_venta=precio,
                cantidad=int(first_item.cantidad),
                proveedor=invoice_data.emisor.nit  # Link to provider by NIT
            )
            print(f"✅ Product payload enriched: {result.payload_product.nombre}")
        
        return result
    
    # --- v1.1: Clasificación de intención + extracción y registro de distribuidor ---
    @chat_webservice_api_router_02.post("/api/chat_v2.0")
    async def chat_with_structure_output(self, request: ChatRequestDTO):
        global GOOGLE_API_KEY
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai",  api_key=self.GOOGLE_API_KEY)
        print("=========REQUEST=========")
        print(request)
        print("=========================")
        conversation = self.find_conversation(request.user_id)

        # Registrar el mensaje actual en memoria y construir historial
        user_input = request.message
        
        # Build message content - can be text only or multimodal (text + image/audio)
        message_content = []
        
        # Always add the text message
        message_content.append({"type": "text", "text": user_input})
        
        # Process file if present
        has_image = False
        has_audio = False
        file_base64 = None
        
        if request.file_path and request.mime_type:
            # Read the file from the provided path and convert to base64
            file_base64 = self._read_file_to_base64(request.file_path)
            file_url = f"data:{request.mime_type};base64,{file_base64}"
            
            # Handle image files
            if request.mime_type.startswith("image/"):
                has_image = True
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": file_url}
                })
                print(f"📸 Image received from path: {request.file_path}, MIME type: {request.mime_type}")
            
            # Handle audio files - use 'media' type as per LangChain requirements
            elif request.mime_type.startswith("audio/"):
                has_audio = True
                message_content.append({
                    "type": "media",
                    "mime_type": request.mime_type,
                    "data": file_base64
                })
                print(f"🎤 Audio received from path: {request.file_path}, MIME type: {request.mime_type}")
            
            else:
                print(f"⚠️ Unsupported MIME type: {request.mime_type}")
        
        # Create the human message with content (text or multimodal)
        # If only text, use simple string format for better compatibility with history
        if len(message_content) == 1:
            conversation.append(HumanMessage(content=user_input))
        else:
            conversation.append(HumanMessage(content=message_content))

        
        history_text = self._history_as_text(request.user_id)
        
        # Extract invoice data if image is present (following Colab pattern)
        invoice_data = None
        if has_image:
            print("🔍 Attempting to extract invoice data from image...")
            invoice_data = self._extract_invoice_from_image(llm, message_content)

        # Usar el modelo Pydantic para clasificación de intención
        model_with_structure = llm.with_structured_output(UserIntention)

        # Clasificación de intención basada en el prompt del Colab
        media_context = ""
        if has_image:
            media_context += "\nNOTA: El usuario adjuntó una imagen (posiblemente una factura). Los datos de la imagen se extraerán automáticamente."
        if has_audio:
            media_context += "\nNOTA: El usuario envió un mensaje de audio. Escucha y transcribe el audio, luego clasifica la intención."
        
        classify_instruction = (
            "Eres un asistente de voz para gestión comercial.\n"
            "Clasifica la intención del usuario y extrae los datos mencionados según el schema.\n"
            "Intenciones disponibles: create_provider, create_client, create_product, other, none, bye.\n"
            "- 'create_provider': cuando el usuario quiere crear un proveedor (puede venir de texto, audio o factura)\n"
            "- 'create_client': cuando el usuario quiere crear un cliente\n"
            "- 'create_product': cuando el usuario quiere crear un producto (puede venir de texto, audio o factura)\n"
            "- 'other': conversación casual u otro propósito\n"
            "- 'none': sin intención clara\n"
            "- 'bye': despedida\n"
            f"{media_context}\n\n"
            f"Historial:\n{history_text}\n\n"
            f"Último mensaje del usuario: {user_input}\n"
            "Si hay audio o imagen, analízalos y extrae la información correspondiente. "
            "Si hay audio, incluye la transcripción en 'audio_transcription'."
        )

        # If we have audio or image, pass them to the classification model
        if has_audio or has_image:
            # Build multimodal classification message
            classification_content = [{"type": "text", "text": classify_instruction}]
            
            # Add audio/image from message_content (skip the first item which is the user text)
            for content_item in message_content[1:]:
                classification_content.append(content_item)
            
            classification_message = HumanMessage(content=classification_content)
            result = model_with_structure.invoke([classification_message])
        else:
            # Text-only classification
            result = model_with_structure.invoke(classify_instruction)
        
        if invoice_data:
            print("🔄 Enriching intention with invoice data...")
            result = self._enrich_intention_with_invoice(result, invoice_data)
        
        # Imprimir resultado de detección de intención
        print("=============== INTENTION DETECTION RESULT ===============")
        print(f"User Intention: {result.userintention}")
        print(f"Audio Transcription: {result.audio_transcription if has_audio else 'N/A'}")
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
            "audio_transcription": result.audio_transcription if has_audio else None,
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
            "has_image": has_image,
            "has_audio": has_audio,
            "invoice_data": invoice_data.model_dump() if invoice_data else None,
        }

        