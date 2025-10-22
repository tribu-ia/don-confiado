from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi_utils.cbv import cbv
from typing import Optional, List, Dict, Any
import os
import tempfile

from dotenv import load_dotenv

from ai.enhanced_graphrag_ingest import (
    ingest_with_ontology, 
    ingest_pdf_with_ontology, 
    get_enhanced_job,
    get_ontology_stats,
    clear_ontology_data
)
from ai.enhanced_graphrag_retrieval import (
    search_contexts_enhanced,
    answer_query_enhanced,
    get_entity_relationships,
    get_knowledge_graph_stats
)


graphrag_api_router = APIRouter()


@cbv(graphrag_api_router)
class ChatClase04:
    def __init__(self):
        load_dotenv()
    

    # Enhanced GraphRAG endpoints with ontology support
    
    @graphrag_api_router.post("/api/graphrag/enhanced/ingest")
    async def enhanced_ingest(
        self, 
        pdf: Optional[UploadFile] = File(None), 
        text: Optional[str] = Form(None), 
        title: Optional[str] = Form(None),
        ontology_type: str = Form("football")
    ):
        """
        Enhanced ingestion with ontology-based knowledge extraction
        """
        try:
            if pdf is None and not text:
                raise HTTPException(status_code=400, detail="Provide a PDF or text")

            if pdf is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    data = await pdf.read()
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    job_id = ingest_pdf_with_ontology(
                        tmp_path, 
                        title=title or pdf.filename,
                        ontology_type=ontology_type
                    )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            else:
                job_id = ingest_with_ontology(
                    text, 
                    title=title or "Input Text",
                    ontology_type=ontology_type
                )

            return {
                "ok": True, 
                "job_id": job_id, 
                "status": get_enhanced_job(job_id),
                "ontology_type": ontology_type
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.post("/api/graphrag/enhanced/ask")
    def enhanced_ask(
        self, 
        query: str,
        retrieval_method: str = "hybrid",
        top_k: int = 5,
        use_graphrag: bool = True
    ):
        """
        Enhanced query with multiple retrieval methods
        """
        try:
            from ai.enhanced_graphrag_retrieval import ensure_vector_index
            ensure_vector_index("text_embeddings")
            contexts = search_contexts_enhanced(
                query, 
                top_k=top_k,
                retrieval_method=retrieval_method
            )
            answer = answer_query_enhanced(
                query, 
                contexts,
                use_graphrag=use_graphrag
            )
            return {
                "answer": answer, 
                "contexts": contexts,
                "retrieval_method": retrieval_method,
                "top_k": top_k
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.get("/api/graphrag/enhanced/job/{job_id}")
    def get_enhanced_job_status(self, job_id: str):
        """
        Get enhanced job status
        """
        try:
            status = get_enhanced_job(job_id)
            return {"job_id": job_id, **status}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.get("/api/graphrag/enhanced/stats")
    def get_knowledge_graph_stats_endpoint(self):
        """
        Get comprehensive knowledge graph statistics
        """
        try:
            stats = get_knowledge_graph_stats()
            return {"ok": True, "stats": stats}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.get("/api/graphrag/enhanced/ontology/{ontology_type}/stats")
    def get_ontology_stats_endpoint(self, ontology_type: str):
        """
        Get statistics for a specific ontology
        """
        try:
            stats = get_ontology_stats(ontology_type)
            return {"ok": True, "ontology_type": ontology_type, "stats": stats}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.get("/api/graphrag/enhanced/entity/{entity_name}/relationships")
    def get_entity_relationships_endpoint(
        self, 
        entity_name: str,
        max_hops: int = 2
    ):
        """
        Get relationships for a specific entity
        """
        try:
            relationships = get_entity_relationships(entity_name, max_hops)
            return {
                "entity_name": entity_name,
                "max_hops": max_hops,
                "relationships": relationships
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @graphrag_api_router.delete("/api/graphrag/enhanced/ontology/{ontology_type}/clear")
    def clear_ontology_data_endpoint(self, ontology_type: str):
        """
        Clear all data for a specific ontology
        """
        try:
            success = clear_ontology_data(ontology_type)
            return {
                "ok": success,
                "ontology_type": ontology_type,
                "message": "Data cleared successfully" if success else "Failed to clear data"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


