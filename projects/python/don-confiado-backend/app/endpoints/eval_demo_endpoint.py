"""
Simple Academic RAG Evaluation Endpoint
For testing and demonstration purposes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

# Import our simple evaluation module
from evaluation.simple_rag_metrics import evaluate_rag_simple

# Import existing RAG components
from ai.enhanced_graphrag_retrieval import (
    search_contexts_enhanced,
    answer_query_enhanced
)
from ai.enhanced_graphrag_config import get_chat_model

# Create router
eval_demo_api_router = APIRouter()


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class RAGEvalRequest(BaseModel):
    """Request model for RAG evaluation demo"""
    query: str
    retrieval_method: str = "hybrid"  # vector, cypher, or hybrid
    top_k: int = 5
    use_graphrag: bool = True


class RAGEvalResponse(BaseModel):
    """Response model with answer and metrics"""
    query: str
    answer: str
    evaluation_metrics: Dict[str, Any]
    latency_ms: float
    timestamp: str


# ============================================
# ENDPOINTS
# ============================================

@eval_demo_api_router.post("/api/eval/rag_demo")
async def rag_evaluation_demo(request: RAGEvalRequest) -> RAGEvalResponse:
    """
    Academic Demo Endpoint: RAG with Evaluation Metrics
    
    This endpoint demonstrates the complete RAG pipeline with evaluation:
    1. Retrieve contexts (vector/cypher/hybrid)
    2. Generate answer
    3. Evaluate quality (Context Relevance, Faithfulness, Answer Relevance)
    
    Example curl:
    ```bash
    curl -X POST http://127.0.0.1:8000/api/eval/rag_demo \
      -H "Content-Type: application/json" \
      -d '{
        "query": "¿Cuáles son los productos más consumidos?",
        "retrieval_method": "hybrid",
        "top_k": 5
      }'
    ```
    """
    
    from datetime import datetime
    
    start_time = time.time()
    
    try:
        # Step 1: Retrieve contexts
        contexts = search_contexts_enhanced(
            query_text=request.query,
            top_k=request.top_k,
            retrieval_method=request.retrieval_method
        )
        
        if not contexts:
            raise HTTPException(status_code=404, detail="No contexts found for query")
        
        # Step 2: Generate answer
        answer = answer_query_enhanced(
            query_text=request.query,
            contexts=contexts,
            use_graphrag=request.use_graphrag
        )
        
        # Step 3: Evaluate RAG quality
        llm = get_chat_model()
        evaluation_metrics = evaluate_rag_simple(
            query=request.query,
            contexts=contexts,
            answer=answer,
            llm=llm
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return RAGEvalResponse(
            query=request.query,
            answer=answer,
            evaluation_metrics=evaluation_metrics,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in RAG evaluation: {str(e)}")


@eval_demo_api_router.get("/api/eval/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Evaluation Demo",
        "endpoints": [
            "POST /api/eval/rag_demo - Full RAG with evaluation",
            "GET /api/eval/health - This endpoint"
        ]
    }


@eval_demo_api_router.get("/api/eval/metrics_info")
async def metrics_info():
    """Information about evaluation metrics"""
    return {
        "metrics": {
            "context_relevance": {
                "description": "Did we retrieve RELEVANT information?",
                "range": "0.0 - 1.0",
                "target": ">= 0.7",
                "measures": "Quality of retrieval (vector/cypher/hybrid)"
            },
            "faithfulness": {
                "description": "Is the answer GROUNDED in context? (No hallucinations)",
                "range": "0.0 - 1.0",
                "target": ">= 0.8",
                "measures": "% of claims supported by retrieved context"
            },
            "answer_relevance": {
                "description": "Does the answer ACTUALLY ANSWER the question?",
                "range": "1 - 5",
                "target": ">= 4",
                "measures": "Completeness, clarity, relevance"
            },
            "overall_quality": {
                "description": "Composite quality score",
                "range": "0.0 - 1.0",
                "calculation": "Weighted: Context(20%) + Faithfulness(40%) + AnswerRel(40%)"
            }
        },
        "quality_thresholds": {
            "excellent": ">= 0.8",
            "good": "0.7 - 0.8",
            "acceptable": "0.5 - 0.7",
            "poor": "< 0.5"
        }
    }
