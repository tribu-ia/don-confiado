from langchain_core.tools import tool


@tool
def mock_supabase_query_tool(params: dict) -> dict:
    """Return mocked Supabase analytics for the report."""
    period = (params or {}).get("period", "last_30_days")
    return {
        "orders": 42,
        "revenue": 12345.67,
        "top_products": [
            {"sku": "SKU-123", "name": "Widget A", "units": 120},
            {"sku": "SKU-456", "name": "Widget B", "units": 95},
        ],
        "period": period,
    }


@tool
def mock_neo4j_query_tool(query: str, params: dict | None = None) -> list[dict]:
    """Return mocked Neo4j graph query results (top customers and relations)."""
    # The inputs are accepted for API compatibility; the output is static for tests.
    return [
        {"topCustomer": "Acme Corp", "purchases": 17, "lastPurchaseDaysAgo": 5},
        {"topCustomer": "Globex LLC", "purchases": 12, "lastPurchaseDaysAgo": 12},
    ]


