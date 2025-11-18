"""
Neo4j Natural Language Query Tool

Uses VectorCypherRetriever from enhanced_graphrag_retrieval to allow
natural language queries that are automatically converted to Cypher.
"""

import os
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from dotenv import load_dotenv
import logging

load_dotenv()

# Try to import from enhanced_graphrag_retrieval
ENHANCED_RETRIEVAL_AVAILABLE = False
NEO4J_GRAPHRAG_AVAILABLE = False
search_with_cypher_retriever = None
search_with_vector_retriever = None
search_with_hybrid_retriever = None

try:
    from ai.enhanced_graphrag_retrieval import (
        search_with_cypher_retriever,
        search_with_vector_retriever,
        search_with_hybrid_retriever,
        NEO4J_GRAPHRAG_AVAILABLE
    )
    ENHANCED_RETRIEVAL_AVAILABLE = True
except ImportError as e:
    logging.debug(f"Could not import enhanced_graphrag_retrieval: {e}")
    ENHANCED_RETRIEVAL_AVAILABLE = False
    NEO4J_GRAPHRAG_AVAILABLE = False
except Exception as e:
    logging.debug(f"Error importing enhanced_graphrag_retrieval: {e}")
    ENHANCED_RETRIEVAL_AVAILABLE = False

# Fallback to direct query tool
from ai.tools.neo4j_tools import neo4j_query_tool


@tool
def neo4j_natural_language_query(
    query_text: str,
    top_k: int = 5,
    retrieval_method: str = "cypher",
    query_type: str = "re_hops"
) -> List[Dict[str, Any]]:
    """
    Query Neo4j using natural language. The retriever automatically converts
    your question into appropriate Cypher queries and returns relevant results.
    
    This tool uses VectorCypherRetriever which:
    1. Embeds your natural language query
    2. Finds similar chunks in the graph
    3. Traverses relationships to gather context
    4. Returns structured results
    
    Args:
        query_text: Natural language question (e.g., "What are the top products consumed?")
        top_k: Number of results to return (default: 5)
        retrieval_method: "cypher", "vector", or "hybrid" (default: "cypher")
        query_type: Query pattern type - "basic", "multi_hop", "entity_relationships", "re_hops" (default: "re_hops")
    
    Returns:
        List of dictionaries with retrieved context and metadata.
        Each result contains:
        - type: retrieval method used
        - content: retrieved information
        - score: similarity score (if available)
        - query_type: query pattern used (if available)
    
    Example:
        result = neo4j_natural_language_query.invoke({
            "query_text": "What products are most consumed by customers?",
            "top_k": 10,
            "retrieval_method": "cypher"
        })
    """
    # Check if enhanced retrieval is available
    if not ENHANCED_RETRIEVAL_AVAILABLE or not NEO4J_GRAPHRAG_AVAILABLE:
        logging.warning(
            "Enhanced retrieval not available. Falling back to direct Cypher queries. "
            "Install neo4j-graphrag and configure OPENAI_API_KEY for natural language queries."
        )
        # Fallback: Try to generate a Cypher query from natural language using LLM
        return _fallback_natural_language_query(query_text, top_k)
    
    # Check for OPENAI_API_KEY (required for neo4j-graphrag)
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning(
            "OPENAI_API_KEY not configured. Falling back to direct Cypher queries. "
            "Configure OPENAI_API_KEY to use natural language queries."
        )
        return _fallback_natural_language_query(query_text, top_k)
    
    try:
        # Use the enhanced retrieval methods
        if retrieval_method == "vector":
            results = search_with_vector_retriever(
                query_text=query_text,
                top_k=top_k
            )
        elif retrieval_method == "hybrid":
            results = search_with_hybrid_retriever(
                query_text=query_text,
                top_k=top_k
            )
        else:  # Default to cypher
            results = search_with_cypher_retriever(
                query_text=query_text,
                top_k=top_k,
                query_type=query_type
            )
        
        # Format results for consistency
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.get("content", str(result)),
                "type": result.get("type", retrieval_method),
                "score": result.get("score"),
                "query_type": result.get("query_type")
            })
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"Natural language query failed: {e}", exc_info=True)
        # Fallback to direct query
        return _fallback_natural_language_query(query_text, top_k)


def _fallback_natural_language_query(query_text: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Fallback: Use LLM to generate Cypher query from natural language,
    then execute it using the direct query tool.
    """
    try:
        from langchain.chat_models import init_chat_model
        
        # Get LLM for query generation
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logging.error("GOOGLE_API_KEY not available for fallback query generation")
            return []
        
        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=google_api_key)
        
        # Generate Cypher query from natural language
        prompt = f"""You are a Neo4j Cypher query expert. Convert the following natural language question into a Cypher query.

Important context about the database:
- Node labels: Consumidor (Customer), Producto (Product), __Entity__ (Entity)
- Relationships: CONSUMIR (consumes), CONTIENE (contains), PERTENECE_A_CATEGORIA (belongs to category)
- Properties may vary: use COALESCE(name, nombre, tipo, category) for names

Natural language question: "{query_text}"

Generate a Cypher query that answers this question. Return ONLY the Cypher query, no explanations.
The query should:
1. Use Spanish labels (Consumidor, Producto)
2. Handle variable property names with COALESCE
3. Return results in a clear format
4. Limit results appropriately

Cypher query:"""
        
        response = llm.invoke(prompt)
        cypher_query = getattr(response, "content", str(response)).strip()
        
        # Remove markdown code blocks if present
        if cypher_query.startswith("```"):
            lines = cypher_query.split("\n")
            cypher_query = "\n".join(lines[1:-1]) if len(lines) > 2 else cypher_query
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        
        logging.info(f"Generated Cypher query: {cypher_query}")
        
        # Execute the generated query
        result = neo4j_query_tool.invoke({
            "query": cypher_query,
            "params": {}
        })
        
        return [{
            "content": result,
            "type": "fallback_llm_generated",
            "cypher_query": cypher_query
        }]
        
    except Exception as e:
        logging.error(f"Fallback query generation failed: {e}", exc_info=True)
        return []

