from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
"""Chat endpoints sin utilizar helpers de memoria de LangChain.

Se usa un almacenamiento en memoria simple (dict + listas) por usuario
para construir el contexto de conversación y se generan prompts como cadenas.
"""

from endpoints.dto.message_dto import (ChatRequestDTO)
from supabase import create_client, Client

# --- Configuración de entorno ---
load_dotenv()


# --- Router y clase del servicio de chat ---
chat_webservice_api_router = APIRouter()

# Memoria en proceso por usuario: { user_id: [ {"role": "human"|"ai", "content": str }, ... ] }
_memory_store = {}


def _get_history(user_id: str):
    if user_id not in _memory_store:
        _memory_store[user_id] = []
    return _memory_store[user_id]


def _append_message(user_id: str, role: str, content: str) -> None:
    history = _get_history(user_id)
    history.append({"role": role, "content": content})


def _history_as_text(user_id: str) -> str:
    lines = []
    for msg in _get_history(user_id):
        if msg.get("role") == "human":
            lines.append(f"Usuario: {msg.get('content', '')}")
        elif msg.get("role") == "ai":
            lines.append(f"Asistente: {msg.get('content', '')}")
    return "\n".join(lines)


@cbv(chat_webservice_api_router)
class ChatWebService:
    # --- v1.0: Chat con memoria en sesión ---
    @chat_webservice_api_router.post("/api/chat_v1.0")
    async def chat_with_memory(self, request: ChatRequestDTO):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
            os.environ["GOOGLE_API_KEY"] = api_key

        # Modelo y prompt del sistema
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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

        # Construcción de historial y prompt como texto
        history_text = _history_as_text(request.user_id)
        user_input = request.message
        _append_message(request.user_id, "human", user_input)

        prompt_text = (
            f"{system_prompt}\n\n"
            f"Historial:\n{history_text}\n\n"
            f"Usuario: {user_input}\n"
            f"Asistente:"
        )

        # Respuesta final directa del modelo
        result = llm.invoke(prompt_text)
        reply = getattr(result, "content", str(result))
        _append_message(request.user_id, "ai", reply)

        return {
            "reply": reply,
        }

    # --- v1.1: Clasificación de intención + extracción y registro de distribuidor ---
    @chat_webservice_api_router.post("/api/chat_v1.1")
    async def chat_with_structure_output(self, request: ChatRequestDTO):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            api_key = input("Por favor, ingrese su API KEY de Google (GOOGLE_API_KEY): ")
            os.environ["GOOGLE_API_KEY"] = api_key

        # Modelo base
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Registrar el mensaje actual en memoria y construir historial
        user_input = request.message
        _append_message(request.user_id, "human", user_input)
        history_text = _history_as_text(request.user_id)

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

        model_with_structure = llm.with_structured_output(intention_schema)

        # Clasificación de intención (prompt plano)
        classify_text = (
            "Eres un clasificador. Lee la conversación y clasifica la intención "
            "estrictamente en una de dos etiquetas: 'Create_distribuitor' u 'Other'. "
            "Usa 'Create_distribuitor' cuando el usuario pretende crear/registrar un proveedor/"
            "distribuidor (p. ej., menciona crear un proveedor/distribuidor). En otro caso usa 'Other'.\n\n"
            f"Historial:\n{history_text}\n\n"
            f"Último mensaje del usuario: {user_input}"
        )

        result = model_with_structure.invoke(classify_text)
        print(result)
        user_intention = result[0]["args"].get("userintention")

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

            # Usar memoria propia y construir prompt plano
            history_text = _history_as_text(request.user_id)
            user_input = request.message

            prompt_text = (
                f"{system_prompt}\n\n"
                f"Historial:\n{history_text}\n\n"
                f"Usuario: {user_input}\n"
                f"Asistente:"
            )

            ai_result = llm.invoke(prompt_text)
            reply = getattr(ai_result, "content", str(ai_result))
            _append_message(request.user_id, "ai", reply)
            print(ai_result)
            return {
                "userintention": "Other",
                "reply": reply,
            }
        else:
            # Rama 'Create_distribuitor': validar completitud y luego extraer datos

            # 1) Verificación de completitud
            completeness_schema = {
                "title": "DistribuidorCompleteness",
                "type": "object",
                "properties": {
                    "is_complete": {"type": "boolean"},
                    "missing_fields": {
                        "type": "array",
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

            completeness_model = llm.with_structured_output(completeness_schema)
            completeness_text = (
                "Evalúa si el mensaje contiene la información completa para crear un distribuidor. "
                "Requisitos: tipo_documento (CC/NIT/CE), numero_documento y (razon_social) o (nombres y apellidos). "
                "Devuelve is_complete=true solo si todos los requisitos están presentes en el mensaje. "
                "Si falta algo, lista los campos faltantes en missing_fields.") + f"\n\nMensaje del usuario: {request.message}"

            completeness = completeness_model.invoke(completeness_text)
            print(completeness)
            is_complete = bool(completeness[0]["args"].get("is_complete", False))
            missing_fields = completeness[0]["args"].get("missing_fields", []) or []

            if not is_complete:
                # Solicitud de datos faltantes (prompt plano + memoria)
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

                reply_obj = llm.invoke(request_missing_text)
                reply_text = getattr(reply_obj, "content", str(reply_obj))
                _append_message(request.user_id, "ai", reply_text)

                return {
                    "userintention": "Create_distribuitor",
                    "status": "need_more_data",
                    "missing_fields": missing_fields,
                    "reply": reply_text,
                }

            # 2) Extracción de datos (solo cuando está completo)
            extraction_schema = {
                "title": "DistribuidorData",
                "description": (
                    "Extra unicamente los campos que el usuario proporciona. No inventes valores."
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

            extractor = llm.with_structured_output(extraction_schema)
            extract_text = (
                "Extrae los campos del distribuidor desde el mensaje del usuario. No inventes datos. "
                "Si un campo no está presente, omítelo (no devuelvas null).\n\n"
                f"Mensaje del usuario: {request.message}"
            )
            extracted_payload = extractor.invoke(extract_text)
            print(extracted_payload)
            extracted = extracted_payload[0]["args"] if isinstance(extracted_payload, list) else extracted_payload
            tipo_documento = extracted.get("tipo_documento")
            numero_documento = extracted.get("numero_documento")
            razon_social = extracted.get("razon_social")
            nombres = extracted.get("nombres")
            apellidos = extracted.get("apellidos")

            # Sanitizar registro (evitar null, vacíos y "null")
            def _valid_value(value: object) -> bool:
                if value is None:
                    return False
                text = str(value).strip()
                if text == "":
                    return False
                if text.lower() == "null":
                    return False
                return True

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
            global _supabase_client
            try:
                _supabase_client
            except NameError:
                _supabase_client = create_client(supabase_url, supabase_key)

            record = {k: v for k, v in extracted.items() if _valid_value(v)}
            record["tipo_tercero"] = "proveedor"

            try:
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
                reply_obj = llm.invoke(confirm_text)
                reply_text = getattr(reply_obj, "content", str(reply_obj))
                _append_message(request.user_id, "ai", reply_text)

                return {
                    "userintention": "Create_distribuitor",
                    "status": "created",
                    "data": data,
                    "reply": reply_text,
                }
            except Exception as e:
                # Manejo de error al crear distribuidor (prompt plano)
                user_input = request.message

                error_text = (
                    "ROLE: Don Confiado, asesor empresarial empático.\n"
                    "Informa que ocurrió un error al crear el distribuidor y que intente de nuevo, "
                    "sin detalles técnicos.\n\n"
                    f"Usuario: {user_input}\n"
                    f"Asistente:"
                )
                reply_obj = llm.invoke(error_text)
                reply_text = getattr(reply_obj, "content", str(reply_obj))
                _append_message(request.user_id, "ai", reply_text)
                return {
                    "userintention": "Create_distribuitor",
                    "status": "error",
                    "error": str(e),
                    "reply": reply_text,
                    "extracted": extracted,
                }
