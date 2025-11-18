import os
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_core.tools import tool
from datetime import datetime, timedelta
import logging
from business.common.connection import SessionLocal

load_dotenv()


def _parse_period(period: str) -> Tuple[datetime, datetime]:
    """
    Parse period string to date range.
    
    Args:
        period: "last_30_days", "last_7_days", "last_90_days", etc.
    
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    
    if period == "last_7_days":
        days = 7
    elif period == "last_30_days":
        days = 30
    elif period == "last_90_days":
        days = 90
    elif period == "last_365_days" or period == "last_year":
        days = 365
    else:
        # Default to 30 days
        days = 30
    
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


@tool
def supabase_query_tool(params: dict) -> dict:
    """
    Query PostgreSQL database (via direct SQL connection) for analytics data.
    Uses the same connection pattern as the rest of the codebase.
    
    Args:
        params: Dict with query parameters:
            - period: "last_30_days", "last_7_days", etc. (default: "last_30_days")
            - Optional: custom filters, date ranges, etc.
    
    Returns:
        Dict with analytics data:
            - orders: int (count of orders/transactions)
            - revenue: float (total revenue)
            - top_products: List[Dict] (top products by sales/quantity)
            - period: str (the period queried)
    """
    period = params.get("period", "last_30_days")
    session: Optional[Session] = None
    
    try:
        start_date, end_date = _parse_period(period)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        session = SessionLocal()
        orders_count = 0
        revenue = 0.0
        top_products = []
        
        # Try to query orders/transactions data
        # Attempt multiple table name variations (whitelist for security)
        # Note: Table names are from a whitelist, so f-string is safe here
        table_names = ["orders", "ventas", "transacciones", "facturas"]
        
        for table_name in table_names:
            try:
                # Try to get orders count and revenue
                # Using whitelisted table names only (security: prevents SQL injection)
                result = session.execute(text(f"""
                    SELECT 
                        COUNT(*) as count,
                        COALESCE(SUM(total), 0) as revenue
                    FROM {table_name}
                    WHERE fecha_creacion >= :start_date 
                      AND fecha_creacion <= :end_date
                """), {
                    "start_date": start_str,
                    "end_date": end_str
                }).mappings().first()
                
                if result:
                    orders_count = result.get("count", 0) or 0
                    revenue = float(result.get("revenue", 0) or 0)
                    break  # Success, exit loop
            except Exception as e:
                # Table doesn't exist or query failed, try next
                logging.debug(f"Table {table_name} not found or query failed: {e}")
                continue
        
        # Try to get top products
        # First try order_items or venta_items (if they exist)
        item_tables = ["order_items", "venta_items", "factura_items"]
        
        for item_table in item_tables:
            try:
                # Try with date filter if fecha_creacion exists
                result = session.execute(text(f"""
                    SELECT 
                        COALESCE(product_sku, producto_sku, sku) as sku,
                        COALESCE(product_name, producto_nombre, nombre) as name,
                        SUM(COALESCE(quantity, cantidad, 0)) as units
                    FROM {item_table}
                    WHERE fecha_creacion >= :start_date 
                      AND fecha_creacion <= :end_date
                    GROUP BY sku, name
                    ORDER BY units DESC
                    LIMIT 5
                """), {
                    "start_date": start_str,
                    "end_date": end_str
                }).mappings().all()
                
                if result:
                    top_products = [
                        {
                            "sku": str(row.get("sku", "N/A")),
                            "name": str(row.get("name", "N/A")),
                            "units": int(row.get("units", 0) or 0)
                        }
                        for row in result
                    ]
                    break  # Success, exit loop
            except Exception:
                # Try without date filter (table might not have fecha_creacion)
                try:
                    result = session.execute(text(f"""
                        SELECT 
                            COALESCE(product_sku, producto_sku, sku) as sku,
                            COALESCE(product_name, producto_nombre, nombre) as name,
                            SUM(COALESCE(quantity, cantidad, 0)) as units
                        FROM {item_table}
                        GROUP BY sku, name
                        ORDER BY units DESC
                        LIMIT 5
                    """)).mappings().all()
                    
                    if result:
                        top_products = [
                            {
                                "sku": str(row.get("sku", "N/A")),
                                "name": str(row.get("name", "N/A")),
                                "units": int(row.get("units", 0) or 0)
                            }
                            for row in result
                        ]
                        break
                except Exception:
                    continue
        
        # Fallback: get top products by quantity from productos table
        if not top_products:
            try:
                result = session.execute(text("""
                    SELECT sku, nombre as name, cantidad as units
                    FROM productos
                    ORDER BY cantidad DESC
                    LIMIT 5
                """)).mappings().all()
                
                top_products = [
                    {
                        "sku": str(row.get("sku", "N/A")),
                        "name": str(row.get("name", "N/A")),
                        "units": int(row.get("units", 0) or 0)
                    }
                    for row in result
                ]
            except Exception as e:
                logging.warning(f"Could not query productos table: {e}")
        
        return {
            "orders": orders_count,
            "revenue": float(revenue),
            "top_products": top_products,
            "period": period,
        }
    except Exception as e:
        # Log error and return empty structure for fail-safe behavior
        logging.error(f"Database query failed: {e}", exc_info=True)
        return {
            "orders": 0,
            "revenue": 0.0,
            "top_products": [],
            "period": period,
        }
    finally:
        if session:
            session.close()

