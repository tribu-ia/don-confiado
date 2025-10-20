import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from neo4j import GraphDatabase

from ai.graphrag_config import (
    get_embeddings,
    get_chat_model,
    get_enhanced_retrieval_query,
)


load_dotenv()


def _get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def ensure_vector_index(index_name: str = "chunk_embeddings") -> int:
    # Determine embedding dims from the embedder config
    # Google text-embedding-004 is 768 dims
    dims = 768

    from business.common.neo4j_connection import ensure_vector_index as _ensure
    _ensure(index_name=index_name, dimensions=dims, similarity="cosine")
    return dims


def search_contexts(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    # Embed the query
    embeddings = get_embeddings()
    q_vec = embeddings.embed_query(query_text)

    driver = _get_driver()
    try:
        with driver.session() as session:
            # Vector search + subquery for enriched context
            cypher = (
                """
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $q_vec)
                YIELD node, score
                CALL {
                  WITH node
                """
                + get_enhanced_retrieval_query()
                + """
                }
                RETURN info, score
                """
            )
            records = session.run(cypher, {"top_k": top_k, "q_vec": q_vec})
            results = []
            for rec in records:
                info = rec.get("info")
                score = rec.get("score")
                if info:
                    results.append({"score": score, **info})
            return results
    finally:
        driver.close()


def answer_query(query_text: str, contexts: List[Dict[str, Any]]) -> str:
    llm = get_chat_model()
    context_text = "\n\n".join([
        "Chunk:\n" + (c.get("chunk") or "") +
        "\nEntities: " + ", ".join([e.get("name", "?") for e in c.get("entities", [])]) +
        "\nRelationships: " + ", ".join([r.get("type", "?") for r in c.get("relationships", [])])
        for c in contexts
    ])

    prompt = (
        "Contexto (no inventes fuera de esto):\n" + context_text +
        "\n\nPregunta: " + query_text +
        "\n\nResponde en 2–4 frases en español, con precisión y citando hechos del contexto."
    )
    result = llm.invoke([{"role": "user", "content": prompt}])
    return getattr(result, "content", str(result))


