# Standard library imports
import os
import uuid
from datetime import datetime

# Third-party imports
from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client

# Local imports
from endpoints.dto.message_dto import ChatRequestDTO
from business.dao.producto_dao import ProductoDAO
from business.dao.tercero_dao import TerceroDAO
from business.entities.producto import Producto
from business.entities.tercero import Tercero
from business.common.connection import SessionLocal
from ai.schemas.facturas import FacturaColombiana, UserIntention

"""
Chat endpoints without using LangChain memory helpers.

Uses simple in-memory storage (dict + lists) per user to build conversation context
and generates prompts as strings.
"""


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


# =============================================================================
# ROUTER AND CLASS DEFINITION
# =============================================================================

chat_webservice_api_router_02 = APIRouter()


@cbv(chat_webservice_api_router_02)
class ChatWebService02:
    """
    Chat service for Don Confiado AI assistant.
    
    Handles multimodal conversations (text, audio, images) with intention detection
    and automatic data extraction from invoices and user inputs.
    """
    
    def __init__(self):
        """Initialize the chat service with environment variables and conversation storage."""
        load_dotenv()
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self._conversations = {}

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
        if conversation_id in self._conversations.keys():
            return self._conversations[conversation_id]
        else:
            conversation = []
            conversation.append(SystemMessage(content=DONCONFIADO_SYSTEM_PROMPT))
            self._conversations[conversation_id] = conversation
            return conversation 

    def _history_as_text(self, user_id: str) -> str:
        """
        Convert conversation history to text format for context.
        
        Args:
            user_id: User identifier for the conversation
            
        Returns:
            Formatted text representation of the conversation history
        """
        lines = []
        conversation = self.find_conversation(user_id)
        for msg in conversation:
            if isinstance(msg, HumanMessage):
                lines.append(f"Usuario: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Asistente: {msg.content}")
        return "\n".join(lines)
    
    # =============================================================================
    # DATA PERSISTENCE METHODS
    # =============================================================================
    
    def _save_product(self, payload):
        """
        Save a product to the database using the extracted payload.
        
        Args:
            payload: PayloadCreateProduct with the extracted product data
            
        Returns:
            Saved Product entity
            
        Raises:
            HTTPException: If there's an error saving the product
        """
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
        """
        Save a tercero (provider or client) to the database using the extracted payload.
        
        Args:
            payload: PayloadCreateProvider or PayloadCreateClient with the extracted data
            tipo_tercero: Either 'proveedor' or 'cliente'
            
        Returns:
            Saved Tercero entity
            
        Raises:
            HTTPException: If there's an error saving the tercero
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
    
    # =============================================================================
    # AI PROCESSING METHODS
    # =============================================================================
    
    def _extract_invoice_from_image(self, llm, message_content):
        """
        Extract invoice data from an image using structured output.
        
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
        """
        Enrich the detected intention with invoice data if applicable.
        
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
    
    # =============================================================================
    # MAIN CHAT ENDPOINT
    # =============================================================================
    
    @chat_webservice_api_router_02.post("/api/chat_v2.0")
    async def chat_with_structure_output(self, request: ChatRequestDTO):
        """
        Main chat endpoint with intention detection and multimodal support.
        
        Handles text, audio, and image inputs with automatic data extraction
        and entity creation (products, providers, clients).
        
        Args:
            request: ChatRequestDTO with user message and optional file data
            
        Returns:
            Dict with chat response, detected intention, and saved entities
        """
        # Initialize LLM
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai", api_key=self.GOOGLE_API_KEY)
        
        print("=========REQUEST=========")
        print(request)
        print("=========================")
        
        # Get or create conversation
        conversation = self.find_conversation(request.user_id)
        user_input = request.message
        
        # Process multimodal content
        message_content, has_image, has_audio = self._process_multimodal_content(request)
        
        # Add message to conversation
        if len(message_content) == 1:
            conversation.append(HumanMessage(content=user_input))
        else:
            conversation.append(HumanMessage(content=message_content))
        
        # Extract invoice data if image is present
        invoice_data = None
        if has_image:
            print("🔍 Attempting to extract invoice data from image...")
            invoice_data = self._extract_invoice_from_image(llm, message_content)
        
        # Classify user intention
        result = self._classify_user_intention(llm, request.user_id, user_input, message_content, has_image, has_audio)
        
        # Enrich intention with invoice data
        if invoice_data:
            print("🔄 Enriching intention with invoice data...")
            result = self._enrich_intention_with_invoice(result, invoice_data)
        
        # Log intention detection results
        self._log_intention_results(result, has_audio)
        
        # Save entities based on detected intention
        saved_entities = self._save_entities_from_intention(result)
        
        # Generate AI response
        try:
            ai_result = llm.invoke(conversation)
            reply = getattr(ai_result, "content", str(ai_result))
        except Exception as e:
            # If multimodal conversation fails, fall back to text-only
            print(f"⚠️ Multimodal response failed, using text-only: {str(e)}")
            fallback_messages = [
                SystemMessage(content=DONCONFIADO_SYSTEM_PROMPT),
                HumanMessage(content=user_input)
            ]
            ai_result = llm.invoke(fallback_messages)
            reply = getattr(ai_result, "content", str(ai_result))
        conversation.append(AIMessage(content=reply))
        
        print("===REPLY===")
        print(reply)
        
        # Return comprehensive response
        return self._build_response(result, reply, saved_entities, has_image, has_audio, invoice_data)
    
    # =============================================================================
    # HELPER METHODS FOR MAIN ENDPOINT
    # =============================================================================
    
    def _process_multimodal_content(self, request: ChatRequestDTO):
        """
        Process multimodal content from the request.
        
        Args:
            request: ChatRequestDTO with optional file data
            
        Returns:
            Tuple of (message_content, has_image, has_audio)
        """
        message_content = [{"type": "text", "text": request.message}]
        has_image = False
        has_audio = False
        
        if request.file_base64 and request.mime_type:
            file_url = f"data:{request.mime_type};base64,{request.file_base64}"
            
            if request.mime_type.startswith("image/"):
                has_image = True
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": file_url}
                })
                print(f"📸 Image received with MIME type: {request.mime_type}")
            
            elif request.mime_type.startswith("audio/"):
                has_audio = True
                # Use media format as supported by LangChain Google GenAI
                message_content.append({
                    "type": "media",
                    "data": request.file_base64,
                    "mime_type": request.mime_type
                })
                print(f"🎤 Audio received: {request.mime_type}")
            
            else:
                print(f"⚠️ Unsupported MIME type: {request.mime_type}")
        
        return message_content, has_image, has_audio
    
    def _classify_user_intention(self, llm, user_id: str, user_input: str, message_content: list, has_image: bool, has_audio: bool):
        """
        Classify user intention using structured output.
        
        Args:
            llm: Language model instance
            user_id: User identifier for conversation history
            user_input: User's text input
            message_content: List with multimodal content (text, audio, image)
            has_image: Whether image is present
            has_audio: Whether audio is present
            
        Returns:
            UserIntention object with classified intention and extracted data
        """
        model_with_structure = llm.with_structured_output(UserIntention)
        
        # Build media context
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
            f"Historial:\n{self._history_as_text(user_id)}\n\n"
            f"Último mensaje del usuario: {user_input}\n"
            "Si hay audio o imagen, analízalos y extrae la información correspondiente. "
            "Si hay audio, incluye la transcripción en 'audio_transcription'."
        )
        
        # Use multimodal classification if needed
        if has_audio or has_image:
            # Build classification message with text instruction + media content
            classification_content = [{"type": "text", "text": classify_instruction}]
            
            # Add audio/image from message_content (skip the first text item)
            for content_item in message_content[1:]:
                classification_content.append(content_item)
            
            classification_message = HumanMessage(content=classification_content)
            try:
                return model_with_structure.invoke([classification_message])
            except Exception as e:
                # Fallback to text-only if multimodal fails
                print(f"⚠️ Multimodal classification failed: {str(e)}")
                return model_with_structure.invoke(classify_instruction)
        else:
            # Text-only classification
            return model_with_structure.invoke(classify_instruction)
    
    def _log_intention_results(self, result, has_audio: bool):
        """Log the intention detection results for debugging."""
        print("=============== INTENTION DETECTION RESULT ===============")
        print(f"User Intention: {result.userintention}")
        print(f"Audio Transcription: {result.audio_transcription if has_audio else 'N/A'}")
        print(f"Payload Provider: {result.payload_provider}")
        print(f"Payload Client: {result.payload_client}")
        print(f"Payload Product: {result.payload_product}")
        print(f"Full Result: {result}")
        print("=========================================================")
    
    def _save_entities_from_intention(self, result):
        """
        Save entities based on detected intention.
        
        Args:
            result: UserIntention object with detected intention and payloads
            
        Returns:
            Dict with saved entities and success status
        """
        saved_entities = {
            'product': {'saved': False, 'entity': None},
            'provider': {'saved': False, 'entity': None},
            'client': {'saved': False, 'entity': None}
        }
        
        # Handle create_product intention
        if result.userintention == "create_product" and result.payload_product:
            try:
                saved_product = self._save_product(result.payload_product)
                saved_entities['product'] = {'saved': True, 'entity': saved_product}
                print(f"🎉 Product '{saved_product.nombre}' saved with SKU: {saved_product.sku}")
            except Exception as e:
                print(f"⚠️ Failed to save product: {str(e)}")
        
        # Handle create_provider intention
        if result.userintention == "create_provider" and result.payload_provider:
            try:
                saved_provider = self._save_tercero(result.payload_provider, 'proveedor')
                saved_entities['provider'] = {'saved': True, 'entity': saved_provider}
                print(f"🎉 Provider '{saved_provider.razon_social}' saved with ID: {saved_provider.id}")
            except Exception as e:
                print(f"⚠️ Failed to save provider: {str(e)}")
        
        # Handle create_client intention
        if result.userintention == "create_client" and result.payload_client:
            try:
                saved_client = self._save_tercero(result.payload_client, 'cliente')
                saved_entities['client'] = {'saved': True, 'entity': saved_client}
                print(f"🎉 Client '{saved_client.razon_social}' saved with ID: {saved_client.id}")
            except Exception as e:
                print(f"⚠️ Failed to save client: {str(e)}")
        
        return saved_entities
    
    def _build_response(self, result, reply: str, saved_entities: dict, has_image: bool, has_audio: bool, invoice_data):
        """
        Build the final response dictionary.
        
        Args:
            result: UserIntention object
            reply: AI-generated response
            saved_entities: Dict with saved entities
            has_image: Whether image was processed
            has_audio: Whether audio was processed
            invoice_data: Extracted invoice data if any
            
        Returns:
            Dict with complete response data
        """
        return {
            "userintention": result.userintention,
            "reply": reply,
            "audio_transcription": result.audio_transcription if has_audio else None,
            "payload_provider": result.payload_provider.model_dump() if result.payload_provider else None,
            "payload_client": result.payload_client.model_dump() if result.payload_client else None,
            "payload_product": result.payload_product.model_dump() if result.payload_product else None,
            "invoice_data": invoice_data.model_dump() if invoice_data else None,
        }

        