from fastapi import APIRouter, HTTPException
from fastapi_utils.cbv import cbv
from typing import List, Tuple, Dict, Any
import os
import json
from dotenv import load_dotenv
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

# LangChain / Google GenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Local imports
from endpoints.dto.message_dto import ChatRequestDTO
from business.common.connection import SessionLocal, engine


chat_clase_03_api_router = APIRouter()


# Constants
# Google GenAI embeddings expect fully-qualified model id: "models/text-embedding-004"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # 768-dim as of Google GenAI
EMBEDDING_DIM = 768 #dimensiones del embedding
CHAT_MODEL_NAME = "gemini-2.5-flash"


DONCONFIADO_RAG_SYSTEM = (
    "Eres Don Confiado, asesor empresarial. Usa estrictamente el contexto recuperado "
    "para responder en español, con tono claro y cercano, en 2–4 frases. "
    "Si la información no está en el contexto, dilo sin inventar."
)

# se encarga de asegurar que la extension pg_vector esté instalada en supabase
def _ensure_pgvector_extension(session: Session) -> None:
    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

# creacion de las tablas de vector de productos
def _create_vector_tables(session: Session) -> None:
    session.execute(text(f"""
        CREATE TABLE IF NOT EXISTS productos_vec (
            id BIGSERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES productos(id) ON DELETE CASCADE,
            content TEXT,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (source_id)
        );
    """))
    session.execute(text(f"""
        CREATE INDEX IF NOT EXISTS idx_productos_vec_embedding
        ON productos_vec USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
    """))

    # Proveedores (distribuidores) con segmentación por chunks
    session.execute(text(f"""
        CREATE TABLE IF NOT EXISTS proveedores_vec (
            id BIGSERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES terceros(id) ON DELETE CASCADE,
            chunk_index INTEGER DEFAULT 0,
            content TEXT,
            embedding vector({EMBEDDING_DIM}),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (source_id, chunk_index)
        );
    """))
    session.execute(text(f"""
        CREATE INDEX IF NOT EXISTS idx_proveedores_vec_embedding
        ON proveedores_vec USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
    """))

    # Clientes: eliminado

# Chunking: Dividir textos largos en partes ("chunks") más pequeñas y solapadas facilita el procesamiento y la búsqueda semántica.
# El sobrelapamiento ("overlap") entre chunks asegura contexto suficiente entre segmentos consecutivos.
def _chunk_text(text_value: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    if not text_value:
        return []
    text_value = text_value.strip()
    if not text_value:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text_value)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text_value[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# Convierte una lista de valores float a un string SQL válido para el vector
def _arr_to_sql_vector(values: List[float]) -> str:
    # Build ARRAY[...]::vector(D)
    floats = ",".join(f"{v:.8f}" for v in values)
    return f"ARRAY[{floats}]::vector({EMBEDDING_DIM})"

# Convierte una lista de textos a una lista de vectores
def _embed_texts(emb: GoogleGenerativeAIEmbeddings, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return emb.embed_documents(texts)

# Construye el contenido del producto
def _build_product_content(row: Dict[str, Any]) -> str:
    proveedor_nombre = row.get("proveedor_nombre") or ""
    proveedor_desc = (
        f"Proveedor: {proveedor_nombre} (id {row.get('proveedor_id')})"
        if row.get("proveedor_id")
        else "Proveedor: N/A"
    )
    return (
        f"Producto: {row.get('nombre')}. SKU: {row.get('sku')}. "
        f"Precio: {row.get('precio_venta')}. Cantidad: {row.get('cantidad')}. "
        f"{proveedor_desc}"
    )

# Construye el contenido del tercero
def _build_tercero_content(row: Dict[str, Any]) -> str:
    nombre = row.get("razon_social") or (
        (row.get("nombres") or "") + " " + (row.get("apellidos") or "")
    ).strip()
    return (
        f"Nombre: {nombre}. Documento: {row.get('tipo_documento')} {row.get('numero_documento')}. "
        f"Tel: {row.get('telefono_celular') or row.get('telefono_fijo') or ''}. "
        f"Dirección: {row.get('direccion') or ''}. Email: {row.get('email') or row.get('email_facturacion') or ''}."
    )

# clase que se encarga de la configuración y el procesamiento de los datos
# se incluye el decorador cbv para que se pueda usar como un endpoint de fastapi con las rutas definidas en el router 
@cbv(chat_clase_03_api_router)
class ChatClase03:
    def __init__(self):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY no configurada")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=self.google_api_key,
        )

    # se encarga de asegurar que la extension pg_vector esté instalada en supabase y crear las tablas de vector de productos
    @chat_clase_03_api_router.post("/api/setup_pgvector")
    def setup_pgvector(self):
        session = SessionLocal()
        try:
            _ensure_pgvector_extension(session)
            _create_vector_tables(session)
            session.commit()
            return {"ok": True, "message": "PGVector y tablas creadas"}
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            session.close()

    # se encarga de sincronizar los embeddings de los productos, proveedores y clientes
    @chat_clase_03_api_router.post("/api/sync_embeddings")
    def sync_embeddings(self):
        session = SessionLocal()
        try:
            # Productos (join con terceros para obtener el nombre del proveedor)
            productos = session.execute(text(
                """
                SELECT p.id,
                       p.sku,
                       p.nombre,
                       p.precio_venta,
                       p.cantidad,
                       p.proveedor_id,
                       COALESCE(t.razon_social, (COALESCE(t.nombres, '') || ' ' || COALESCE(t.apellidos, ''))) AS proveedor_nombre
                FROM productos p
                LEFT JOIN terceros t ON t.id = p.proveedor_id
                """
            )).mappings().all()
            prod_texts = [_build_product_content(p) for p in productos]
            prod_vecs = _embed_texts(self.embeddings, prod_texts)
            for row, vec, content in zip(productos, prod_vecs, prod_texts):
                sql = text(
                    f"""
                    INSERT INTO productos_vec (source_id, content, embedding, metadata)
                    VALUES (:source_id, :content, { _arr_to_sql_vector(vec) }, :metadata)
                    ON CONFLICT (source_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW();
                    """
                ).bindparams(bindparam("metadata", type_=JSONB))
                metadata_payload = {
                    "proveedor_id": row.get("proveedor_id"),
                    "proveedor_nombre": row.get("proveedor_nombre"),
                }
                session.execute(sql, {"source_id": row["id"], "content": content, "metadata": metadata_payload})

            # Proveedores (terceros tipo proveedor) con segmentación
            proveedores = session.execute(text(
                """
                SELECT id, tipo_documento, numero_documento, razon_social, nombres, apellidos,
                       telefono_fijo, telefono_celular, direccion, email, email_facturacion
                FROM terceros WHERE tipo_tercero = 'proveedor'
                """
            )).mappings().all()
            for prov in proveedores:
                base_text = _build_tercero_content(prov)
                chunks = _chunk_text(base_text, chunk_size=600, overlap=100)
                if not chunks:
                    chunks = [base_text]
                chunk_vecs = _embed_texts(self.embeddings, chunks)
                for idx, (chunk, vec) in enumerate(zip(chunks, chunk_vecs)):
                    sql = text(
                        f"""
                        INSERT INTO proveedores_vec (source_id, chunk_index, content, embedding, metadata)
                        VALUES (:source_id, :chunk_index, :content, { _arr_to_sql_vector(vec) }, '{{}}'::jsonb)
                        ON CONFLICT (source_id, chunk_index) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            created_at = NOW();
                        """
                    )
                    session.execute(sql, {"source_id": prov["id"], "chunk_index": idx, "content": chunk})

            # Clientes: eliminado

            session.commit()
            return {"ok": True, "message": "Embeddings sincronizados"}
        except Exception as e:
            session.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            session.close()

    # se encarga de buscar el contexto relevante para la pregunta del usuario
    # <->: indica que se está usando la distancia euclidiana (L2) para medir la similitud entre el vector de la pregunta y el vector de los productos, proveedores y clientes
    # <=>: indica que se está usando la distancia coseno para medir la similitud entre el vector de la pregunta y el vector de los productos, proveedores y clientes
    # <#>: indica que se está usando la distancia del producto punto para medir la similitud entre el vector de la pregunta y el vector de los productos, proveedores y clientes
    def _search_context(self, session: Session, query_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q_vec = self.embeddings.embed_query(query_text)
        q_vec_sql = _arr_to_sql_vector(q_vec)
        results = session.execute(text(
            f"""
            WITH q AS (SELECT {q_vec_sql} AS embedding)
            SELECT 'producto' AS source, pv.source_id, pv.content, (pv.embedding <-> q.embedding) AS distance
            FROM productos_vec pv, q
            UNION ALL
            SELECT 'proveedor' AS source, pr.source_id, pr.content, (pr.embedding <-> q.embedding) AS distance
            FROM proveedores_vec pr, q
            -- clientes_vec eliminado
            ORDER BY distance ASC
            LIMIT :k
            """
        ), {"k": top_k}).mappings().all()
        return [dict(r) for r in results]

    def _build_rag_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> List[HumanMessage]:
        context_text = "\n\n".join(f"[{r['source']}] {r['content']}" for r in contexts)
        user_text = (
            "Contexto recuperado (no inventes fuera de esto):\n" + context_text +
            "\n\nPregunta del usuario: " + question +
            "\n\nResponde en 2–4 frases, en español, claras y accionables."
        )
        return [
            SystemMessage(content=DONCONFIADO_RAG_SYSTEM),
            HumanMessage(content=user_text),
        ]

    @chat_clase_03_api_router.post("/api/chat_clase_03")
    def chat_rag(self, request: ChatRequestDTO):
        session = SessionLocal()
        try:
            # Retrieve relevant context
            contexts = self._search_context(session, request.message, top_k=8)

            # Initialize chat model
            llm = init_chat_model(CHAT_MODEL_NAME, model_provider="google_genai", api_key=self.google_api_key)
            messages = self._build_rag_prompt(request.message, contexts)
            ai_result = llm.invoke(messages)
            reply = getattr(ai_result, "content", str(ai_result))

            # This reply can be forwarded to WhatsApp by the TS service
            return {
                "reply": reply,
                "contexts": contexts,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            session.close()


