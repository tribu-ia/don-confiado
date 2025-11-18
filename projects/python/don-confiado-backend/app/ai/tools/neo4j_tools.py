import os
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from business.common.neo4j_connection import get_neo4j_driver


@tool
def neo4j_query_tool(query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query against Neo4j and return results.
    
    Args:
        query: Cypher query string
        params: Optional parameters dict for parameterized queries
    
    Returns:
        List of dictionaries, each representing a result record.
        Returns empty list on error to allow workflow continuation.
    """
    driver = None
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run(query, params or {})
            records = []
            for record in result:
                # Convert Neo4j record to dict
                records.append(dict(record))
            return records
    except Exception as e:
        # Log error but don't crash - return empty list for fail-safe behavior
        # The workflow can continue with partial data
        import logging
        logging.error(f"Neo4j query failed: {e}", exc_info=True)
        return []
    finally:
        if driver:
            try:
                driver.close()
            except Exception:
                pass


