import os
from dotenv import load_dotenv
from typing import List

# LangChain Google GenAI for consistency with class 03
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model


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


# Ontology for market research (modifiable)
def get_entities() -> List[str]:
    return [
        "Product",
        "Category",
        "Brand",
        "Region",
        "Retailer",
        "TimePeriod",
        "Metric",
        "Promotion",
    ]


def get_relations() -> List[str]:
    return [
        "BELONGS_TO_CATEGORY",
        "BRAND_OF",
        "TOP_SELLS_IN",
        "TREND_IN",
        "SOLD_BY",
        "MEASURED_AS",
        "PROMOTED_IN",
        "RELATED_TO",
    ]


def get_extraction_prompt() -> str:
    return (
        """
Extract entities and relationships from the text below.

ENTITIES: Product, Category, Brand, Region, Retailer, TimePeriod, Metric, Promotion
RELATIONSHIPS: BELONGS_TO_CATEGORY, BRAND_OF, TOP_SELLS_IN, TREND_IN, SOLD_BY, MEASURED_AS, PROMOTED_IN, RELATED_TO

Rules:
1. Use exact entity type names and relationship types
2. Only extract explicit information from the text
3. Return valid JSON only

Text: {text}

Return JSON:
{
  "entities": [{"name": "...", "type": "...", "description": "..."}],
  "relationships": [{"start": "...", "type": "...", "end": "..."}]
}
"""
    ).strip()


# Retrieval cypher (enhanced pattern summarized)
def get_enhanced_retrieval_query() -> str:
    return (
        """
WITH node AS chunk

OPTIONAL MATCH (e1:__Entity__)-[:FROM_CHUNK]->(chunk)
OPTIONAL MATCH (e1)-[r]-(e2:__Entity__)
WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK']
OPTIONAL MATCH (e2)-[:FROM_CHUNK]->(c2:Chunk)
OPTIONAL MATCH (chunk)-[:FROM_DOCUMENT]->(d1:Document)
OPTIONAL MATCH (c2)-[:FROM_DOCUMENT]->(d2:Document)

WITH chunk,
     collect(DISTINCT e1) AS entities,
     collect(DISTINCT r) AS rels,
     collect(DISTINCT c2) AS related_chunks,
     collect(DISTINCT d1) + collect(DISTINCT d2) AS docs

RETURN {
  chunk: chunk.text,
  entities: [e IN entities WHERE e IS NOT NULL | {name: coalesce(e.name, e.id), type: e.type}],
  relationships: [r IN rels WHERE r IS NOT NULL | {start: coalesce(startNode(r).name, startNode(r).id), type: type(r), end: coalesce(endNode(r).name, endNode(r).id)}],
  sources: [c IN related_chunks WHERE c IS NOT NULL | c.text],
  documents: [d IN docs WHERE d IS NOT NULL | coalesce(d.title, d.path, d.id)]
} AS info
"""
    ).strip()


