from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi_utils.cbv import cbv
from typing import Optional
import os
import tempfile

from dotenv import load_dotenv

from business.common.neo4j_connection import verify_connection
from ai.graphrag_retrieval import ensure_vector_index, search_contexts, answer_query
from ai.graphrag_ingest import ingest_text, ingest_pdf, get_job


graphrag_api_router = APIRouter()


@cbv(graphrag_api_router)
class ChatClase04:
    def __init__(self):
        load_dotenv()
        # If the user provided explicit env via notebook, keep as is
        # Otherwise, rely on .env

    @graphrag_api_router.post("/api/graphrag/setup_neo4j")
    def setup_neo4j(self):
        try:
            ok = verify_connection()
            if not ok:
                raise HTTPException(status_code=500, detail="Neo4j connection failed")
            dims = ensure_vector_index("chunk_embeddings")
            return {"ok": True, "dimensions": dims}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.post("/api/graphrag/ingest")
    async def ingest(self, pdf: Optional[UploadFile] = File(None), text: Optional[str] = Form(None), title: Optional[str] = Form(None)):
        try:
            if pdf is None and not text:
                raise HTTPException(status_code=400, detail="Provide a PDF or text")

            if pdf is not None:
                # Save to temp file for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    data = await pdf.read()
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    job_id = ingest_pdf(tmp_path, title=title or pdf.filename)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                job_id = ingest_text(text, title=title or "Input Text")

            return {"ok": True, "job_id": job_id, "status": get_job(job_id)}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.post("/api/graphrag/ask")
    def ask(self, query: str):
        try:
            ensure_vector_index("chunk_embeddings")
            contexts = search_contexts(query, top_k=3)
            answer = answer_query(query, contexts)
            return {"answer": answer, "contexts": contexts}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


