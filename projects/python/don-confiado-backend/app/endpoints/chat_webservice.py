# Import necessary tools from FastAPI and other libraries
from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
"""Chat endpoints sin utilizar helpers de memoria de LangChain.
Se usa un almacenamiento en memoria simple (dict + listas) por usuario
para construir el contexto de conversación y se generan prompts como cadenas.
"""

# Import a data structure (DTO) for chat requests
from endpoints.dto.message_dto import (ChatRequestDTO)
from supabase import create_client, Client

# --- Configuración de entorno ---
# Load environment variables from a .env file. This is where sensitive information like API keys are stored.
load_dotenv()


# --- Router y clase del servicio de chat ---
# Create an API router. This helps organize different parts of your API.
chat_webservice_api_router = APIRouter()

# Memoria en proceso por usuario: { user_id: [ {"role": "human"|"ai", "content": str }, ... ] }
# This is a simple way to store conversation history for each user.
# It's a dictionary where each key is a user's ID, and the value is a list of messages.
_memory_store = {}


def _get_history(user_id: str):
 # This function gets the conversation history for a specific user.
 # If the user doesn't have a history yet, it creates an empty one.
    if user_id not in _memory_store:
        _memory_store[user_id] = []
    return _memory_store[user_id]


def _append_message(user_id: str, role: str, content: str) -> None:
 # This function adds a new message to a user's conversation history.
 # 'role' tells us if the message is from the 'human' (user) or 'ai' (assistant).
    history = _get_history(user_id)
    history.append({"role": role, "content": content})


def _history_as_text(user_id: str) -> str:
 # This function takes the conversation history and turns it into a single block of text.
 # This text is then used to give the AI context for its next response.
    lines = []
    for msg in _get_history(user_id):
        if msg.get("role") == "human":
            lines.append(f"Usuario: {msg.get('content', '')}")
        elif msg.get("role") == "ai":
            lines.append(f"Asistente: {msg.get('content', '')}")
    return "\n".join(lines)


@cbv(chat_webservice_api_router)
class ChatWebService:
 # This class defines the different chat services (endpoints) available.
    # --- v1.0: Chat con memoria en sesión ---
    @chat_webservice_api_router.post("/api/chat_v1.0")
    async def chat_with_memory(self, request: ChatRequestDTO):
 # This is an API endpoint that handles chat requests.
 # It uses a simple in-memory storage to remember past conversations.
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
 # If the Google API key isn't set, it asks the user to input it.
            api_key = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
            os.environ["GOOGLE_API_KEY"] = api_key

        # Modelo y prompt del sistema
        # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Modelo Original
        llm = ChatGoogleGenerativeAI(model="learnlm-2.0-flash-experimental")
 # This is the main instruction for the AI, telling it what its role is, what it should do, and how it should behave.
 # It sets the personality and rules for "Don Confiado".
        system_prompt = """ROLE:
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

        # Build the conversation history and the full prompt text for the AI.
        history_text = _history_as_text(request.user_id)
        user_input = request.message
 # Add the user's message to the conversation history.
        _append_message(request.user_id, "human", user_input)
 # Combine the system prompt, conversation history, and current user input into one prompt for the AI.

        prompt_text = (
            f"{system_prompt}\n\n"
            f"Historial:\n{history_text}\n\n"
            f"Usuario: {user_input}\n"
            f"Asistente:"
        )

        # Get the AI's response by sending the prompt to the language model.
        result = llm.invoke(prompt_text)
 # Extract the actual text content from the AI's response.
 # 
        reply = getattr(result, "content", str(result))
 # Add the AI's response to the conversation history.
        _append_message(request.user_id, "ai", reply)

        return {
            "reply": reply,
        }

 # To improve performance, consider using a more persistent and scalable memory solution
 # like a database (e.g., Redis, PostgreSQL) instead of in-memory storage, especially for
 # a production environment with many users. This would prevent conversation history from
 # being lost if the server restarts and allow for easier scaling.

    # --- v1.1: Clasificación de intención + extracción y registro de distribuidor ---
    @chat_webservice_api_router.post("/api/chat_v1.1")
    async def chat_with_structure_output(self, request: ChatRequestDTO):
 # This endpoint is more advanced, it tries to understand the user's "intention" (what they want to do).
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
            os.environ["GOOGLE_API_KEY"] = api_key

        # Modelo y prompt del sistema
        # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Modelo Original
        llm = ChatGoogleGenerativeAI(model="learnlm-2.0-flash-experimental")

        # Record the current message in memory and build the history.
        user_input = request.message
        _append_message(request.user_id, "human", user_input)
        history_text = _history_as_text(request.user_id)

 # This defines a "schema" (a blueprint) for what kind of information we expect the AI to return
 # when it tries to classify the user's intention.
        # Esquema de intención + clasificador estructurado
        intention_schema = {
            "title": "UserIntention",
            "description": (
                "Clasifica la intención del mensaje del usuario. "
                "Devuelve solo una de las etiquetas permitidas."
            ),
            "type": "object",
            "properties": {
                "userintention": {
                    "type": "string",
                    "enum": ["Create_distribuitor", "Other"],
                    "description": (
                        "'Create_distribuitor': cuando el usuario quiere crear/registrar un proveedor/distribuidor. "
                        "'Other': conversación casual u otro propósito."
                    ),
                }
            },
            "required": ["userintention"],
            "additionalProperties": False,
        }

        #Let's break it down:
        # llm: This is your base Large Language Model instance, created with ChatGoogleGenerativeAI(...). By default, when you give it a prompt, it gives you back a string of text
        # .with_structured_output(...): This is a LangChain method that "chains" a new capability onto the llm. It takes a schema as an argument and creates a new, modified model.
        # intention_schema: This is a Python dictionary that defines the exact JSON structure you want the AI to produce. In your code, this schema tells the model:
        # The output must be a JSON object.
        #   The object must have one key: "userintention".
        #   The value of "userintention" must be a string.
        #   That string must be either "Create_distribuitor" or "Other".
        # No other keys or values are allowed.
        # What model_with_structure Does:        
        # The resulting model_with_structure is a new, more powerful version of the original llm. When you call model_with_structure.invoke(...):
        #   - It still sends your prompt to the Google AI model.
        #   - But instead of just returning whatever text the model generates, it performs an extra step: it forces the model's output to conform to the intention_schema.
        #   - The final result is not a simple string, but a parsed Python object (like a dictionary) that matches your schema.
        # In simple terms: You are telling the AI, "Don't just answer my question with words. Answer it by filling out this specific form."
        

        model_with_structure = llm.with_structured_output(intention_schema)

        # This is the instruction given to the AI to classify the user's intention.
        classify_text = (
            "Eres un clasificador. Lee la conversación y clasifica la intención "
            "estrictamente en una de dos etiquetas: 'Create_distribuitor' u 'Other'. "
            "Usa 'Create_distribuitor' cuando el usuario pretende crear/registrar un proveedor/"
            "distribuidor (p. ej., menciona crear un proveedor/distribuidor). En otro caso usa 'Other'.\n\n"
            f"Historial:\n{history_text}\n\n"
            f"Último mensaje del usuario: {user_input}"
        )

        result = model_with_structure.invoke(classify_text)
 # Print the classification result for debugging.
        print(result)
        user_intention = result[0]["args"].get("userintention")
 # If the AI determines the intention is "Other", it means the user is just chatting.

        if user_intention == "Other":
            # Rama 'Other': respuesta general con memoria
            system_prompt = """ROLE:
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

            # Use the conversation memory and build a plain prompt for the AI to respond.
            history_text = _history_as_text(request.user_id)
            user_input = request.message

 # Combine the system prompt, history, and user input to generate a conversational response.
 # This is similar to v1.0 but happens after intention classification.
            prompt_text = (
                f"{system_prompt}\n\n"
                f"Historial:\n{history_text}\n\n"
                f"Usuario: {user_input}\n"
                f"Asistente:"
            )

 # Get the AI's response.
            ai_result = llm.invoke(prompt_text)
            reply = getattr(ai_result, "content", str(ai_result))
 # Add the AI's response to the history.
            _append_message(request.user_id, "ai", reply)
            print(ai_result)
            return {
                "userintention": "Other",
                "reply": reply,
            }
       

        # =========================================================================================
        # AGENTIC WORKFLOW: CREATE DISTRIBUTOR
        # =========================================================================================
        # If the initial intent classification determines the user wants to create a
        # distributor, this block executes a multi-step agentic workflow. The process
        # is designed to be robust, ensuring all necessary information is gathered
        # before attempting to save it to the database.
        #
        # This workflow consists of two main phases:
        # 1. Validation Phase: Check if the user's message contains all mandatory data.
        # 2. Extraction Phase: If the data is complete, extract it into a structured format.
        #
        else:
            # ---------------------------------------------------------------------------------
            # PHASE 1: INFORMATION-GATHERING AND VALIDATION
            # ---------------------------------------------------------------------------------
            # The first step is to determine if the user's message is self-contained
            # and provides all the required information to create a new distributor.
            # This prevents incomplete records and reduces back-and-forth conversation.

            # 1.1) Define the "Completeness" Schema
            # This JSON schema forces the AI to act as a validation tool. It must decide
            # if the user's message is complete and explicitly list any missing fields.
            completeness_schema = {
                "title": "DistribuidorCompleteness",
                "description": "Evaluates if the user's message contains all mandatory fields for creating a distributor.",
                "type": "object",
                "properties": {
                    "is_complete": {
                        "type": "boolean",
                        "description": "True if all required fields are present, otherwise False."
                    },
                    "missing_fields": {
                        "type": "array",
                        "description": "A list of required fields that are missing from the user's message.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "tipo_documento",
                                "numero_documento",
                                "razon_social",
                                "nombres",
                                "apellidos"
                            ]
                        }
                    }
                },
                "required": ["is_complete", "missing_fields"],
                "additionalProperties": False,
            }

            # 1.2) Create a Specialized Model for Validation
            # The base LLM is enhanced with the `with_structured_output` method, forcing
            # its response to conform to the `completeness_schema`.
            completeness_model = llm.with_structured_output(completeness_schema)

            # 1.3) Create the Validation Prompt
            # This prompt instructs the AI to perform the evaluation. It clearly defines
            # the business rule: a distributor needs a document type, number, and either
            # a company name (razon_social) or personal name (nombres and apellidos).
            completeness_text = (
                "Evalúa si el mensaje contiene la información completa para crear un distribuidor. "
                "Requisitos: tipo_documento (CC/NIT/CE), numero_documento y (razon_social) o (nombres y apellidos). "
                "Devuelve is_complete=true solo si todos los requisitos están presentes en el mensaje. "
                "Si falta algo, lista los campos faltantes en missing_fields."
            ) + f"\n\nMensaje del usuario: {request.message}"

            # 1.4) Invoke the Model and Parse the Response
            # The model returns a structured object. We extract the `is_complete` boolean
            # and the list of `missing_fields` to guide the next step.
            completeness = completeness_model.invoke(completeness_text)
            print(completeness)
            is_complete = bool(completeness[0]["args"].get("is_complete", False))
            missing_fields = completeness[0]["args"].get("missing_fields", []) or []

            # 1.5) Handle Incomplete Information
            # If the validation fails, the bot must ask the user for the missing data.
            if not is_complete:
                # The AI is prompted to generate a user-friendly question, incorporating
                # the specific fields that were identified as missing.
                history_text = _history_as_text(request.user_id)
                user_input = request.message

                request_missing_text = (
                    "ROLE: Don Confiado, asesor empresarial amable y claro.\n"
                    "Pide al usuario, en una sola oración y sin tecnicismos, los datos faltantes: "
                    f"{', '.join(missing_fields)}.\n\n"
                    f"Historial:\n{history_text}\n\n"
                    f"Usuario: {user_input}\n"
                    f"Asistente:"
                )

                # The standard LLM generates a natural language response.
                reply_obj = llm.invoke(request_missing_text)
                reply_text = getattr(reply_obj, "content", str(reply_obj))
                _append_message(request.user_id, "ai", reply_text)

                # The API returns a specific status indicating that more data is needed,
                # allowing the frontend to handle this state accordingly.
                return {
                    "userintention": "Create_distribuitor",
                    "status": "need_more_data",
                    "missing_fields": missing_fields,
                    "reply": reply_text,
                }

            # ---------------------------------------------------------------------------------
            # PHASE 2: DATA EXTRACTION
            # ---------------------------------------------------------------------------------
            # This phase only runs if the validation in Phase 1 was successful.
            # The goal is to extract all available distributor fields from the user's
            # message into a clean, structured format.

            # 2.1) Define the "Extraction" Schema
            # This schema lists all possible fields for a distributor, including optional
            # ones like phone and email. The AI's task is to fill this "form" using
            # only the information provided in the user's message.
            extraction_schema = {
                "title": "DistribuidorData",
                "description": (
                    "Extrae unicamente los campos que el usuario proporciona. No inventes valores."
                ),
                "type": "object",
                "properties": {
                    "tipo_documento": {
                        "type": "string",
                        "enum": ["CC", "NIT", "CE"],
                        "description": "Tipo de documento: CC, NIT o CE"
                    },
                    "numero_documento": {"type": "string"},
                    "razon_social": {"type": "string"},
                    "nombres": {"type": "string"},
                    "apellidos": {"type": "string"},
                    "telefono_fijo": {"type": "string"},
                    "telefono_celular": {"type": "string"},
                    "direccion": {"type": "string"},
                    "email": {"type": "string"}
                },
                "additionalProperties": False,
            }

            # 2.2) Create a Specialized Model for Extraction
            extractor = llm.with_structured_output(extraction_schema)

            # 2.3) Create the Extraction Prompt
            # This prompt instructs the AI to act as a data-entry assistant, extracting
            # fields and ignoring any that are not explicitly mentioned.
            extract_text = (
                "Extrae los campos del distribuidor desde el mensaje del usuario. No inventes datos. "
                "Si un campo no está presente, omítelo (no devuelvas null).\n\n"
                f"Mensaje del usuario: {request.message}"
            )

            # 2.4) Invoke the Extractor Model
            # The result is a structured payload containing the extracted data.
            extracted_payload = extractor.invoke(extract_text)
            print(extracted_payload)
            extracted = extracted_payload[0]["args"] if isinstance(extracted_payload, list) else extracted_payload
            tipo_documento = extracted.get("tipo_documento")
            numero_documento = extracted.get("numero_documento")
            razon_social = extracted.get("razon_social")
            nombres = extracted.get("nombres")
            apellidos = extracted.get("apellidos")

            # 2.5) Sanitize the Extracted Data
            # A helper function to ensure that no null, empty, or "null" string values
            # are passed to the database insertion logic.
            def _valid_value(value: object) -> bool:
                if value is None:
                    return False
                text = str(value).strip()
                if text == "":
                    return False
                if text.lower() == "null":
                    return False
                return True


 # Check if the necessary Supabase credentials are set in the environment variables.
            # Validación credenciales Supabase
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            if not supabase_url or not supabase_key:
                # Respuesta breve informando falta de credenciales
                user_input = request.message

                creds_text = (
                    "ROLE: Don Confiado, asesor empresarial.\n"
                    "Informa brevemente que faltan las credenciales de Supabase (SUPABASE_URL / "
                    "SUPABASE_SERVICE_ROLE_KEY) y que deben configurarse antes de continuar.\n\n"
                    f"Usuario: {user_input}\n"
                    f"Asistente:"
                )
 # If credentials are missing, the AI informs the user.
                reply_obj = llm.invoke(creds_text)
                reply_text = getattr(reply_obj, "content", str(reply_obj))
                _append_message(request.user_id, "ai", reply_text)
                return {
                    "userintention": "Create_distribuitor",
                    "status": "error",
                    "error": "Missing Supabase credentials",
                    "reply": reply_text,
                    "extracted": extracted,
                }

            # Inicialización cliente Supabase
 # Initialize the Supabase client to interact with the database.
            global _supabase_client
            try:
                _supabase_client
            except NameError:
                _supabase_client = create_client(supabase_url, supabase_key)

 # Filter out invalid values from the extracted data and set the 'tipo_tercero' to 'proveedor'.
            record = {k: v for k, v in extracted.items() if _valid_value(v)}
            record["tipo_tercero"] = "proveedor"

            try:
 # Insert the new distributor record into the 'terceros' table in Supabase.
                # Inserción en Supabase y confirmación
                response = _supabase_client.table("terceros").insert(record).execute()
                data = getattr(response, "data", None)

                user_input = request.message

                confirm_text = (
                    "ROLE: Don Confiado, asesor empresarial.\n"
                    "Confirma brevemente que el distribuidor ha sido creado exitosamente.\n\n"
                    f"Usuario: {user_input}\n"
                    f"Asistente:"
                )
 # The AI confirms that the distributor has been created successfully.
                reply_obj = llm.invoke(confirm_text)
                reply_text = getattr(reply_obj, "content", str(reply_obj))
 # The AI's reply is added to the history.
                _append_message(request.user_id, "ai", reply_text)

                return {
                    "userintention": "Create_distribuitor",
                    "status": "created",
                    "data": data,
                    "reply": reply_text,
                }
            except Exception as e:
 # If there's an error during the creation process, the AI informs the user.
                # Manejo de error al crear distribuidor (prompt plano)
                user_input = request.message

                error_text = (
                    "ROLE: Don Confiado, asesor empresarial empático.\n"
                    "Informa que ocurrió un error al crear el distribuidor y que intente de nuevo, "
                    "sin detalles técnicos.\n\n"
                    f"Usuario: {user_input}\n"
                    f"Asistente:"
                )
 # The AI generates an error message.
                reply_obj = llm.invoke(error_text)
                reply_text = getattr(reply_obj, "content", str(reply_obj))
 # The AI's reply is added to the history.
                _append_message(request.user_id, "ai", reply_text)
                return {
                    "userintention": "Create_distribuitor",
                    "status": "error",
                    "error": str(e),
                    "reply": reply_text,
                    "extracted": extracted,
                }
