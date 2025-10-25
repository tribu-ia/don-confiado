import os
from dotenv import load_dotenv
from typing import List, Dict, Any
import json

# LangChain Google GenAI for consistency with class 03
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model

# Import market research ontology
from ai.market_research_ontology import (
    get_market_research_entities,
    get_market_research_relations,
    get_market_research_extraction_prompt
)

load_dotenv()


# Embedding/LLM factories (Gemini based)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured")
    # Use Google's text-embedding-004 (768 dims)
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )


def get_chat_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured")
    return init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)


# Market Research Ontology (Primary)






def get_original_prompt() -> str:
    """Returns the original generic extraction prompt"""
    return """
You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

Use only the following node and relationship types (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
•⁠  ⁠Do not return any additional information other than the JSON in it.
•⁠  ⁠Omit any backticks around the JSON - simply output the JSON on its own.
•⁠  ⁠The JSON object must not wrapped into a list - it is its own JSON object.
•⁠  ⁠Property names must be enclosed in double quotes

Examples:
{examples}

Input text:

{text}
"""


def get_enhanced_retrieval_queries() -> Dict[str, str]:
    """Returns different retrieval query patterns for different use cases"""
    return {
        "basic": """
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)-[r]->(m)
RETURN chunk, r, m, similarity_score
LIMIT 5
""",
        
        "multi_hop": """
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)-[r]->(child)-[rr]->(child2)
RETURN 
  similarity_score, 
  r,
  rr,
  apoc.map.removeKey(properties(chunk), 'embedding') AS chunk,
  apoc.map.removeKey(properties(child), 'embedding') AS child,
  apoc.map.removeKey(properties(child2), 'embedding') AS child2
ORDER BY similarity_score DESC
""",
        
        "entity_relationships": """
WITH node AS chunk

// Buscar nodos _Entity_ conectados a este chunk
OPTIONAL MATCH (e1:_Entity_)-[:FROM_CHUNK]->(chunk)

// Obtener relaciones entre entidades
OPTIONAL MATCH (e1)-[r]-(e2:_Entity_)
WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK']

// Obtener los chunks de origen de las entidades relacionadas
OPTIONAL MATCH (e2)-[:FROM_CHUNK]->(c2:Chunk)

// Retornar sin incluir la propiedad 'embedding'
RETURN
    chunk { .* , embedding: null } AS chunk,
    [e IN collect(DISTINCT e1) | e { .* , embedding: null }] AS entities_1,
    [e IN collect(DISTINCT e2) | e { .* , embedding: null }] AS entities_2,
    collect(DISTINCT r) AS relationships,
    [c IN collect(DISTINCT c2) | c { .* , embedding: null }] AS related_chunks
""",
        
        "re_hops": """
//1) Go out up to N hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,4}()
UNWIND relList AS rel

//2) collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks,
 collect(DISTINCT rel) AS rels

//3) format and return context
RETURN '=== text ===n' + apoc.text.join([c in chunks | c.text], 'n---n') + 'nn=== kg_rels ===n' +
 apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], 'n---n') AS info
"""
    }


def get_kg_builder_config() -> Dict[str, Any]:
    """Returns configuration for SimpleKGPipeline"""
    return {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "entities": get_market_research_entities_config(),
        "relations": get_market_research_relations_config(),
        "perform_entity_resolution": True,
        "from_pdf": False
    }


def get_retriever_config() -> Dict[str, Any]:
    """Returns configuration for different retrievers"""
    return {
        "vector_retriever": {
            "index_name": "text_embeddings",
            "top_k": 5
        },
        "cypher_retriever": {
            "index_name": "text_embeddings", 
            "query": get_enhanced_retrieval_queries()["re_hops"],
            "top_k": 5
        },
        "hybrid_retriever": {
            "vector_weight": 0.7,
            "cypher_weight": 0.3
        }
    }


# Market Research Ontology Functions
def get_market_research_entities_config() -> List[str]:
    """Get market research entities for knowledge extraction"""
    return get_market_research_entities()


def get_market_research_relations_config() -> List[str]:
    """Get market research relationships for knowledge extraction"""
    return get_market_research_relations()


def get_market_research_extraction_prompt_config() -> str:
    """Get market research extraction prompt"""
    return get_market_research_extraction_prompt()
