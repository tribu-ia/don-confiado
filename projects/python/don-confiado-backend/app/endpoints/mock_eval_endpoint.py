"""
Mock RAG Demo Endpoint - For Academic Testing Without Database
Simulates the RAG pipeline with sample data
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import time
from datetime import datetime

# Import our evaluation module
from evaluation.simple_rag_metrics import evaluate_rag_simple
from ai.enhanced_graphrag_config import get_chat_model

# Create router
mock_eval_api_router = APIRouter()


class MockRAGRequest(BaseModel):
    """Request for mock RAG demo"""
    query: str
    simulation_scenario: str = "good_quality"  # good_quality, hallucination, poor_retrieval


class MockRAGResponse(BaseModel):
    """Response with mocked answer and real evaluation"""
    query: str
    answer: str
    contexts_summary: str
    evaluation_metrics: Dict[str, Any]
    latency_ms: float
    note: str


def _get_mock_contexts(scenario: str) -> List[Dict[str, Any]]:
    """Generate mock retrieved contexts based on scenario"""
    
    if scenario == "good_quality":
        return [
            {
                "type": "hybrid",
                "content": "Los productos más consumidos por familias colombianas incluyen arroz (consumo promedio 3kg/mes), huevos AA x30 (407 unidades vendidas en último mes), y leche entera 1L (386 unidades vendidas). El arroz es un producto básico presente en el 95% de los hogares."
            },
            {
                "type": "hybrid", 
                "content": "Según el estudio de mercado de 2025, los productos lácteos representan el 18% del gasto familiar mensual. La leche entera es el producto lácteo más consumido, seguido por queso y yogurt."
            },
            {
                "type": "hybrid",
                "content": "En la región andina, los patrones de consumo muestran preferencia por productos tradicionales como papa, arroz, y panela. Las familias de estratos 1-3 destinan el 35% de su presupuesto a alimentos básicos."
            }
        ]
    elif scenario == "hallucination":
        # Minimal context to encourage hallucination
        return [
            {
                "type": "vector",
                "content": "Estudio de consumo familiar en Colombia, año 2025."
            }
        ]
    elif scenario == "poor_retrieval":
        # Irrelevant contexts
        return [
            {
                "type": "vector",
                "content": "La industria automotriz en Colombia ha crecido un 12% en el último año."
            },
            {
                "type": "vector",
                "content": "El sector tecnológico ofrece múltiples oportunidades de inversión."
            }
        ]
    else:
        return _get_mock_contexts("good_quality")


def _generate_mock_answer(query: str, contexts: List[Dict], scenario: str) -> str:
    """Generate a mock answer using LLM based on contexts"""
    
    llm = get_chat_model()
    
    context_text = "\n\n".join([ctx["content"] for ctx in contexts])
    
    if scenario == "hallucination":
        # More creative prompt that might hallucinate
        prompt = f"""Responde esta pregunta de forma creativa y detallada:

Pregunta: {query}

Contexto disponible:
{context_text}

Responde en 2-3 frases."""
    else:
        # Strict prompt
        prompt = f"""Responde SOLO basándote en el contexto proporcionado. No inventes información.

Contexto:
{context_text}

Pregunta: {query}

Responde en 2-3 frases usando SOLO información del contexto."""
    
    try:
        result = llm.invoke([{"role": "user", "content": prompt}])
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        # Fallback answer
        return f"Error generando respuesta: {str(e)}"


@mock_eval_api_router.post("/api/eval/mock_demo")
async def mock_rag_demo(request: MockRAGRequest) -> MockRAGResponse:
    """
    Academic Mock RAG Demo - Works without database
    
    Simulates different quality scenarios:
    - good_quality: Good contexts, should score high
    - hallucination: Minimal context, might hallucinate
    - poor_retrieval: Irrelevant contexts, low context relevance
    
    Example:
    ```bash
    curl -X POST http://127.0.0.1:8000/api/eval/mock_demo \
      -H "Content-Type: application/json" \
      -d '{
        "query": "¿Cuáles son los productos más consumidos?",
        "simulation_scenario": "good_quality"
      }' | jq '.'
    ```
    """
    
    start_time = time.time()
    
    # Get mock contexts based on scenario
    contexts = _get_mock_contexts(request.simulation_scenario)
    
    # Generate answer
    answer = _generate_mock_answer(request.query, contexts, request.simulation_scenario)
    
    # Evaluate with our real evaluation system
    llm = get_chat_model()
    evaluation_metrics = evaluate_rag_simple(
        query=request.query,
        contexts=contexts,
        answer=answer,
        llm=llm
    )
    
    latency_ms = (time.time() - start_time) * 1000
    
    contexts_summary = f"{len(contexts)} contexts retrieved ({request.simulation_scenario} scenario)"
    
    return MockRAGResponse(
        query=request.query,
        answer=answer,
        contexts_summary=contexts_summary,
        evaluation_metrics=evaluation_metrics,
        latency_ms=round(latency_ms, 2),
        note=f"This is a MOCK demo for academic purposes. Scenario: {request.simulation_scenario}"
    )


@mock_eval_api_router.get("/api/eval/scenarios")
async def list_scenarios():
    """List available simulation scenarios for testing"""
    return {
        "scenarios": {
            "good_quality": {
                "description": "High quality contexts and answer",
                "expected_metrics": {
                    "context_relevance": ">= 0.8",
                    "faithfulness": ">= 0.9", 
                    "answer_relevance": ">= 4"
                }
            },
            "hallucination": {
                "description": "Minimal context - tests hallucination detection",
                "expected_metrics": {
                    "context_relevance": "~0.5",
                    "faithfulness": "< 0.7 (hallucinations likely)",
                    "answer_relevance": "variable"
                }
            },
            "poor_retrieval": {
                "description": "Irrelevant contexts - tests retrieval quality",
                "expected_metrics": {
                    "context_relevance": "< 0.4",
                    "faithfulness": "variable",
                    "answer_relevance": "< 3"
                }
            }
        },
        "usage": "Set 'simulation_scenario' parameter to one of the scenario names above"
    }
