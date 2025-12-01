"""
Simple Academic RAG Evaluation Metrics
For demonstration purposes - simplified, clear implementations
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional
import json


# ============================================
# METRIC MODELS (Pydantic for validation)
# ============================================

class ContextRelevanceMetrics(BaseModel):
    """Measures: Did we retrieve RELEVANT information?"""
    context_relevance_score: float = Field(ge=0.0, le=1.0, description="0-1 relevance score")
    retrieval_method: str = Field(description="vector, cypher, or hybrid")
    num_contexts_retrieved: int
    relevant_contexts: int
    irrelevant_contexts: int
    quality_label: Literal["poor", "acceptable", "good", "excellent"]


class FaithfulnessMetrics(BaseModel):
    """Measures: Is the answer GROUNDED in retrieved context? (No hallucinations)"""
    faithfulness_score: float = Field(ge=0.0, le=1.0, description="% claims supported")
    total_claims: int
    supported_claims: int
    unsupported_claims: List[str] = Field(default_factory=list)
    has_hallucinations: bool
    hallucination_severity: Literal["none", "low", "medium", "high"]


class AnswerRelevanceMetrics(BaseModel):
    """Measures: Does the answer ACTUALLY ANSWER the question?"""
    answer_relevance_score: int = Field(ge=1, le=5, description="1-5 scale")
    addresses_query: bool
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)
    reasoning: str
    recommendation: Literal["reject", "revise", "accept"]


# ============================================
# EVALUATOR CLASSES
# ============================================

class SimpleContextRelevanceEvaluator:
    """Academic demo: Evaluate if retrieved contexts are relevant to query"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate(self, query: str, contexts: List[Dict[str, Any]]) -> ContextRelevanceMetrics:
        """
        Simple LLM-as-judge evaluation of context relevance
        
        Args:
            query: User's question
            contexts: Retrieved context chunks
            
        Returns:
            ContextRelevanceMetrics with scores
        """
        
        # Simplify contexts for evaluation (first 300 chars each)
        context_summaries = []
        for i, ctx in enumerate(contexts[:5]):  # Max 5 for demo
            content = str(ctx.get('content', ''))[:300]
            context_summaries.append(f"Context {i+1}: {content}")
        
        # Build evaluation prompt
        prompt = f"""Eres un evaluador académico. Determina si estos CONTEXTOS RECUPERADOS son relevantes para la CONSULTA.

CONSULTA DEL USUARIO: "{query}"

CONTEXTOS RECUPERADOS:
{chr(10).join(context_summaries)}

TAREA: Para cada contexto, determina si contiene información útil para responder la consulta.

Responde SOLO en JSON con esta estructura exacta:
{{
    "context_relevance_score": 0.0-1.0,
    "retrieval_method": "{contexts[0].get('type', 'unknown') if contexts else 'unknown'}",
    "num_contexts_retrieved": {len(contexts)},
    "relevant_contexts": X,
    "irrelevant_contexts": Y,
    "quality_label": "poor" | "acceptable" | "good" | "excellent"
}}

Donde:
- context_relevance_score = relevant_contexts / num_contexts_retrieved
- quality_label: poor (<0.5), acceptable (0.5-0.7), good (0.7-0.9), excellent (>0.9)
"""
        
        try:
            # Use structured output for reliable parsing
            structured_llm = self.llm.with_structured_output(ContextRelevanceMetrics)
            result = structured_llm.invoke(prompt)
            return result
        except Exception as e:
            # Fallback: manual evaluation
            print(f"Structured output failed, using fallback: {e}")
            return ContextRelevanceMetrics(
                context_relevance_score=0.5,
                retrieval_method="unknown",
                num_contexts_retrieved=len(contexts),
                relevant_contexts=len(contexts) // 2,
                irrelevant_contexts=len(contexts) // 2,
                quality_label="acceptable"
            )


class SimpleFaithfulnessEvaluator:
    """Academic demo: Detect hallucinations by verifying claims against context"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate(self, answer: str, contexts: List[Dict[str, Any]]) -> FaithfulnessMetrics:
        """
        Simple hallucination detection:
        1. Extract claims from answer
        2. Verify each claim against context
        3. Calculate faithfulness score
        
        Args:
            answer: Generated answer to check
            contexts: Retrieved contexts answer should be based on
            
        Returns:
            FaithfulnessMetrics with hallucination detection
        """
        
        # Step 1: Extract factual claims from answer
        claims_prompt = f"""Extrae SOLO las afirmaciones factuales verificables de esta respuesta.
No incluyas opiniones, solo hechos que puedan ser verificados.

RESPUESTA: "{answer}"

Responde en JSON: {{"claims": ["afirmación 1", "afirmación 2", ...]}}"""
        
        try:
            claims_result = self.llm.invoke(claims_prompt)
            claims_data = json.loads(claims_result.content)
            claims = claims_data.get("claims", [])
        except:
            # Fallback: split by periods
            claims = [s.strip() for s in answer.split('.') if len(s.strip()) > 10][:5]
        
        if not claims:
            # No claims to verify = no hallucinations
            return FaithfulnessMetrics(
                faithfulness_score=1.0,
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
                has_hallucinations=False,
                hallucination_severity="none"
            )
        
        # Step 2: Prepare context for verification
        context_text = "\n\n".join([
            str(ctx.get('content', ''))[:800]  # Max 800 chars per context
            for ctx in contexts[:5]
        ])
        
        # Step 3: Verify each claim
        supported = []
        unsupported = []
        
        for claim in claims:
            verify_prompt = f"""¿Este CONTEXTO apoya esta AFIRMACIÓN?

CONTEXTO:
{context_text}

AFIRMACIÓN A VERIFICAR: "{claim}"

Responde SOLO en JSON: {{"supported": true/false, "reason": "breve explicación"}}"""
            
            try:
                verification = self.llm.invoke(verify_prompt)
                result = json.loads(verification.content)
                
                if result.get("supported", False):
                    supported.append(claim)
                else:
                    unsupported.append(claim)
            except:
                # On error, assume not supported (conservative)
                unsupported.append(claim)
        
        # Step 4: Calculate metrics
        faithfulness_score = len(supported) / len(claims) if claims else 1.0
        has_hallucinations = len(unsupported) > 0
        
        # Determine severity
        if not has_hallucinations:
            severity = "none"
        elif len(unsupported) == 1:
            severity = "low"
        elif len(unsupported) <= len(claims) * 0.3:
            severity = "medium"
        else:
            severity = "high"
        
        return FaithfulnessMetrics(
            faithfulness_score=faithfulness_score,
            total_claims=len(claims),
            supported_claims=len(supported),
            unsupported_claims=unsupported,
            has_hallucinations=has_hallucinations,
            hallucination_severity=severity
        )


class SimpleAnswerRelevanceEvaluator:
    """Academic demo: Evaluate if answer actually answers the question"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate(self, query: str, answer: str) -> AnswerRelevanceMetrics:
        """
        Simple answer relevance check using LLM-as-judge
        
        Args:
            query: Original user question
            answer: Generated answer
            
        Returns:
            AnswerRelevanceMetrics with quality scores
        """
        
        prompt = f"""Evalúa si esta RESPUESTA responde adecuadamente a la CONSULTA del usuario.

CONSULTA: "{query}"

RESPUESTA: "{answer}"

Evalúa en escala 1-5:
- answer_relevance_score: ¿Qué tan relevante es? (1=irrelevante, 5=perfecta)
- completeness: ¿Responde todos los aspectos?
- clarity: ¿Es clara?

También determina:
- addresses_query: true si responde la pregunta, false si no
- reasoning: Breve explicación de tu evaluación

Responde en JSON siguiendo este esquema:
{{
    "answer_relevance_score": 1-5,
    "addresses_query": true/false,
    "completeness": 1-5,
    "clarity": 1-5,
    "reasoning": "tu explicación aquí",
    "recommendation": "accept" | "revise" | "reject"
}}"""
        
        try:
            structured_llm = self.llm.with_structured_output(AnswerRelevanceMetrics)
            result = structured_llm.invoke(prompt)
            
            # Auto-set recommendation if not provided
            if result.answer_relevance_score >= 4:
                result.recommendation = "accept"
            elif result.answer_relevance_score >= 3:
                result.recommendation = "revise"
            else:
                result.recommendation = "reject"
            
            return result
        except Exception as e:
            print(f"Structured output failed, using fallback: {e}")
            # Fallback
            return AnswerRelevanceMetrics(
                answer_relevance_score=3,
                addresses_query=True,
                completeness=3,
                clarity=3,
                reasoning="Evaluación automática fallida - usando valores por defecto",
                recommendation="revise"
            )


# ============================================
# CONVENIENCE FUNCTION FOR DEMO
# ============================================

def evaluate_rag_simple(
    query: str,
    contexts: List[Dict[str, Any]],
    answer: str,
    llm
) -> Dict[str, Any]:
    """
    All-in-one RAG evaluation for academic demo
    
    Returns:
        Dictionary with all three metric types
    """
    
    # Initialize evaluators
    context_eval = SimpleContextRelevanceEvaluator(llm)
    faith_eval = SimpleFaithfulnessEvaluator(llm)
    answer_eval = SimpleAnswerRelevanceEvaluator(llm)
    
    # Run evaluations
    context_metrics = context_eval.evaluate(query, contexts)
    faith_metrics = faith_eval.evaluate(answer, contexts)
    answer_metrics = answer_eval.evaluate(query, answer)
    
    # Calculate overall quality score (weighted average)
    overall_quality = (
        context_metrics.context_relevance_score * 0.2 +  # 20% weight
        faith_metrics.faithfulness_score * 0.4 +          # 40% weight
        (answer_metrics.answer_relevance_score - 1) / 4 * 0.4  # 40% weight (normalized)
    )
    
    return {
        "query": query[:200],
        "answer": answer[:200],
        "context_relevance": {
            "score": context_metrics.context_relevance_score,
            "retrieval_method": context_metrics.retrieval_method,
            "quality": context_metrics.quality_label,
            "relevant_out_of_total": f"{context_metrics.relevant_contexts}/{context_metrics.num_contexts_retrieved}"
        },
        "faithfulness": {
            "score": faith_metrics.faithfulness_score,
            "has_hallucinations": faith_metrics.has_hallucinations,
            "severity": faith_metrics.hallucination_severity,
            "claims_verified": f"{faith_metrics.supported_claims}/{faith_metrics.total_claims}",
            "unsupported_claims": faith_metrics.unsupported_claims
        },
        "answer_relevance": {
            "score": answer_metrics.answer_relevance_score,
            "addresses_query": answer_metrics.addresses_query,
            "completeness": answer_metrics.completeness,
            "clarity": answer_metrics.clarity,
            "recommendation": answer_metrics.recommendation,
            "reasoning": answer_metrics.reasoning
        },
        "overall_quality": round(overall_quality, 3),
        "quality_summary": _get_quality_summary(overall_quality, faith_metrics.has_hallucinations)
    }


def _get_quality_summary(overall_quality: float, has_hallucinations: bool) -> str:
    """Helper to generate quality summary"""
    if has_hallucinations:
        return "⚠️ HALLUCINATION DETECTED - Review required"
    elif overall_quality >= 0.8:
        return "✅ EXCELLENT - High quality response"
    elif overall_quality >= 0.7:
        return "✅ GOOD - Acceptable quality"
    elif overall_quality >= 0.5:
        return "⚠️ ACCEPTABLE - Could be improved"
    else:
        return "❌ POOR - Needs significant improvement"
