import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Import neo4j-graphrag components
try:
    from neo4j_graphrag.retrievers import VectorRetriever, HybridRetriever, VectorCypherRetriever
    from neo4j_graphrag.generation.graphrag import GraphRAG
    from neo4j_graphrag.llm import OpenAILLM as LLM
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
    NEO4J_GRAPHRAG_AVAILABLE = True
except ImportError:
    NEO4J_GRAPHRAG_AVAILABLE = False

from ai.enhanced_graphrag_config import (
    get_embeddings,
    get_chat_model,
    get_enhanced_retrieval_queries,
    get_retriever_config
)

load_dotenv()


def _get_driver():
    """Get Neo4j driver connection"""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def _setup_neo4j_graphrag_components():
    """Setup LLM and embeddings for neo4j-graphrag"""
    if not NEO4J_GRAPHRAG_AVAILABLE:
        raise RuntimeError("neo4j-graphrag not available. Install with: pip install neo4j-graphrag")
    
    # Use OpenAI for neo4j-graphrag compatibility
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not configured for neo4j-graphrag")
    
    llm = LLM(
        model_name="gpt-4o-mini",
        model_params={
            "response_format": {"type": "json_object"},
            "temperature": 0
        }
    )
    
    embedder = Embeddings()
    
    return llm, embedder


def ensure_vector_index(index_name: str = "text_embeddings") -> int:
    """Ensure vector index exists and return dimensions"""
    # OpenAI embeddings are 1536 dimensions
    dims = 1536
    
    from business.common.neo4j_connection import ensure_vector_index as _ensure
    _ensure(index_name=index_name, dimensions=dims, similarity="cosine")
    return dims


def search_with_vector_retriever(
    query_text: str, 
    top_k: int = 5,
    index_name: str = "text_embeddings"
) -> List[Dict[str, Any]]:
    """
    Search using VectorRetriever (basic vector similarity)
    """
    if not NEO4J_GRAPHRAG_AVAILABLE:
        raise RuntimeError("neo4j-graphrag not available")
    
    driver = _get_driver()
    try:
        llm, embedder = _setup_neo4j_graphrag_components()
        
        vector_retriever = VectorRetriever(
            driver=driver,
            index_name=index_name,
            embedder=embedder
        )
        
        results = vector_retriever.get_search_results(
            query_text=query_text,
            top_k=top_k
        )
        
        return [{"type": "vector", "score": getattr(record, "score", 0.0), "content": str(record)} 
                for record in results.records]
        
    finally:
        driver.close()


def search_with_cypher_retriever(
    query_text: str,
    top_k: int = 5,
    index_name: str = "text_embeddings",
    query_type: str = "re_hops"
) -> List[Dict[str, Any]]:
    """
    Search using VectorCypherRetriever with custom Cypher queries
    """
    if not NEO4J_GRAPHRAG_AVAILABLE:
        raise RuntimeError("neo4j-graphrag not available")
    
    driver = _get_driver()
    try:
        llm, embedder = _setup_neo4j_graphrag_components()
        
        # Get the appropriate retrieval query
        queries = get_enhanced_retrieval_queries()
        cypher_query = queries.get(query_type, queries["re_hops"])
        
        vector_cypher_retriever = VectorCypherRetriever(
            driver=driver,
            index_name=index_name,
            cypher_query=cypher_query,
            embedder=embedder
        )
        
        results = vector_cypher_retriever.get_search_results(
            query_text=query_text,
            top_k=top_k
        )
        
        return [{"type": "cypher", "query_type": query_type, "content": str(record)} 
                for record in results.records]
        
    finally:
        driver.close()


def search_with_hybrid_retriever(
    query_text: str,
    top_k: int = 5,
    index_name: str = "text_embeddings",
    vector_weight: float = 0.7,
    cypher_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Search using HybridRetriever (combines vector and cypher)
    """
    if not NEO4J_GRAPHRAG_AVAILABLE:
        raise RuntimeError("neo4j-graphrag not available")
    
    driver = _get_driver()
    try:
        llm, embedder = _setup_neo4j_graphrag_components()
        
        # Create individual retrievers
        vector_retriever = VectorRetriever(
            driver=driver,
            index_name=index_name,
            embedder=embedder
        )
        
        queries = get_enhanced_retrieval_queries()
        cypher_retriever = VectorCypherRetriever(
            driver=driver,
            index_name=index_name,
            cypher_query=queries["re_hops"],
            embedder=embedder
        )
        
        # Create hybrid retriever
        hybrid_retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            cypher_retriever=cypher_retriever,
            vector_weight=vector_weight,
            cypher_weight=cypher_weight
        )
        
        results = hybrid_retriever.get_search_results(
            query_text=query_text,
            top_k=top_k
        )
        
        return [{"type": "hybrid", "content": str(record)} 
                for record in results.records]
        
    finally:
        driver.close()


def search_contexts_enhanced(
    query_text: str, 
    top_k: int = 5,
    retrieval_method: str = "hybrid"
) -> List[Dict[str, Any]]:
    """
    Enhanced context search with multiple retrieval methods
    """
    try:
        if retrieval_method == "vector":
            return search_with_vector_retriever(query_text, top_k)
        elif retrieval_method == "cypher":
            return search_with_cypher_retriever(query_text, top_k)
        elif retrieval_method == "hybrid":
            return search_with_hybrid_retriever(query_text, top_k)
        else:
            # Default to hybrid
            return search_with_hybrid_retriever(query_text, top_k)
    except Exception as e:
        print(f"Error in enhanced retrieval: {e}")
        # Fallback to basic vector search
        return search_with_vector_retriever(query_text, top_k)


def answer_query_enhanced(
    query_text: str, 
    contexts: List[Dict[str, Any]],
    use_graphrag: bool = True
) -> str:
    """
    Enhanced query answering with GraphRAG support
    """
    if use_graphrag and NEO4J_GRAPHRAG_AVAILABLE:
        try:
            driver = _get_driver()
            llm, embedder = _setup_neo4j_graphrag_components()
            
            # Use GraphRAG for enhanced reasoning
            graphrag = GraphRAG(
                driver=driver,
                llm=llm,
                embedder=embedder
            )
            
            # Generate answer using GraphRAG
            result = graphrag.generate(
                query=query_text,
                context=contexts
            )
            
            return result.content if hasattr(result, 'content') else str(result)
            
        except Exception as e:
            print(f"GraphRAG error, falling back to basic LLM: {e}")
            # Fallback to basic LLM
            pass
    
    # Basic LLM fallback
    llm = get_chat_model()
    context_text = "\n\n".join([
        f"Context {i+1}:\n{context.get('content', '')}" 
        for i, context in enumerate(contexts)
    ])
    
    prompt = (
        f"Contexto (no inventes fuera de esto):\n{context_text}\n\n"
        f"Pregunta: {query_text}\n\n"
        f"Responde en 2–4 frases en español, con precisión y citando hechos del contexto."
    )
    
    result = llm.invoke([{"role": "user", "content": prompt}])
    return getattr(result, "content", str(result))


def get_entity_relationships(
    entity_name: str,
    max_hops: int = 2
) -> Dict[str, Any]:
    """
    Get relationships for a specific entity
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            query = f"""
            MATCH (e:__Entity__ {{name: $entity_name}})
            OPTIONAL MATCH path = (e)-[r*1..{max_hops}]-(related:__Entity__)
            RETURN 
                e as entity,
                collect(DISTINCT path) as paths,
                collect(DISTINCT r) as relationships
            """
            
            result = session.run(query, entity_name=entity_name)
            record = result.single()
            
            if record:
                return {
                    "entity": dict(record["entity"]),
                    "paths": [dict(path) for path in record["paths"] if path],
                    "relationships": [dict(rel) for rel in record["relationships"] if rel]
                }
            else:
                return {"entity": None, "paths": [], "relationships": []}
                
    finally:
        driver.close()


def get_knowledge_graph_stats() -> Dict[str, Any]:
    """
    Get comprehensive statistics about the knowledge graph
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            # Entity type distribution
            entity_query = """
            MATCH (n:__Entity__)
            RETURN labels(n) as entity_type, count(n) as count
            ORDER BY count DESC
            """
            
            # Relationship type distribution
            rel_query = """
            MATCH ()-[r]->()
            WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK']
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
            """
            
            # Document and chunk counts
            doc_query = """
            MATCH (d:Document)
            OPTIONAL MATCH (d)<-[:FROM_DOCUMENT]-(c:Chunk)
            RETURN count(d) as doc_count, count(c) as chunk_count
            """
            
            entity_results = session.run(entity_query)
            rel_results = session.run(rel_query)
            doc_results = session.run(doc_query)
            
            entities = [{"type": record["entity_type"], "count": record["count"]} 
                      for record in entity_results]
            relationships = [{"type": record["rel_type"], "count": record["count"]} 
                           for record in rel_results]
            doc_record = doc_results.single()
            
            return {
                "entities": entities,
                "relationships": relationships,
                "total_entities": sum(e["count"] for e in entities),
                "total_relationships": sum(r["count"] for r in relationships),
                "documents": doc_record["doc_count"] if doc_record else 0,
                "chunks": doc_record["chunk_count"] if doc_record else 0
            }
            
    finally:
        driver.close()
