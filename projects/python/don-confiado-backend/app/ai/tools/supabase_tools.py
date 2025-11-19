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
                # Use DATE() function for proper date comparison (ignores time component)
                result = session.execute(text(f"""
                    SELECT 
                        COUNT(*) as count,
                        COALESCE(SUM(total), 0) as revenue
                    FROM {table_name}
                    WHERE DATE(fecha_creacion) >= DATE(:start_date)
                      AND DATE(fecha_creacion) <= DATE(:end_date)
                """), {
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().first()
                
                # Check if we got a valid result with data
                if result is not None:
                    count_val = result.get("count")
                    if count_val is not None and int(count_val) > 0:
                        orders_count = int(count_val)
                        revenue = float(result.get("revenue", 0) or 0)
                        logging.info(f"Found {orders_count} orders in {table_name} with revenue ${revenue:,.2f}")
                        break  # Success, exit loop
            except Exception as e:
                # Table doesn't exist or query failed, try next
                logging.warning(f"Table {table_name} query failed: {e}")
                session.rollback()  # Rollback on error to allow next query
                continue
        
        # Try to get top products
        # First try order_items or venta_items (if they exist)
        item_tables = ["order_items", "venta_items", "factura_items"]
        
        for item_table in item_tables:
            try:
                # Try with date filter if fecha_creacion exists
                # Use DATE() function for proper date comparison
                # Try different column name combinations based on table
                # venta_items uses producto_sku/producto_nombre
                # order_items might use product_sku/product_name
                result = session.execute(text(f"""
                    SELECT 
                        producto_sku as sku,
                        producto_nombre as name,
                        SUM(cantidad) as units
                    FROM {item_table}
                    WHERE DATE(fecha_creacion) >= DATE(:start_date)
                      AND DATE(fecha_creacion) <= DATE(:end_date)
                    GROUP BY producto_sku, producto_nombre
                    ORDER BY units DESC
                    LIMIT 5
                """), {
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().all()
                
                if result and len(result) > 0:
                    top_products = [
                        {
                            "sku": str(row.get("sku", "N/A")),
                            "name": str(row.get("name", "N/A")),
                            "units_sold": int(row.get("units", 0) or 0)
                        }
                        for row in result
                    ]
                    logging.info(f"Found {len(top_products)} top products from {item_table}")
                    break  # Success, exit loop
            except Exception as e:
                logging.warning(f"Query failed for {item_table} with date filter: {e}")
                session.rollback()  # Rollback to allow next query
                # Try without date filter or with different column names
                try:
                    # Try with producto_sku/producto_nombre (Spanish schema)
                    result = session.execute(text(f"""
                        SELECT 
                            producto_sku as sku,
                            producto_nombre as name,
                            SUM(cantidad) as units
                        FROM {item_table}
                        GROUP BY producto_sku, producto_nombre
                        ORDER BY units DESC
                        LIMIT 5
                    """)).mappings().all()
                    
                    if result and len(result) > 0:
                        top_products = [
                            {
                                "sku": str(row.get("sku", "N/A")),
                                "name": str(row.get("name", "N/A")),
                                "units_sold": int(row.get("units", 0) or 0)
                            }
                            for row in result
                        ]
                        logging.info(f"Found {len(top_products)} top products from {item_table} (no date filter)")
                        break
                except Exception as e2:
                    logging.warning(f"Fallback query also failed for {item_table}: {e2}")
                    session.rollback()  # Rollback to allow next table
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
        
        # Calculate additional metrics
        avg_order_value = float(revenue) / orders_count if orders_count > 0 else 0.0
        
        # Calculate revenue, costs, and profitability metrics per product
        for product in top_products:
            # Try to get revenue, costs, and prices for this product from venta_items
            try:
                product_metrics_result = session.execute(text("""
                    SELECT 
                        SUM(subtotal) as product_revenue,
                        SUM(costo_unitario * cantidad) as total_cost,
                        AVG(precio_unitario) as avg_price,
                        AVG(costo_unitario) as avg_cost
                    FROM venta_items
                    WHERE producto_sku = :sku
                      AND DATE(fecha_creacion) >= DATE(:start_date)
                      AND DATE(fecha_creacion) <= DATE(:end_date)
                """), {
                    "sku": product.get("sku"),
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().first()
                
                if product_metrics_result:
                    product["revenue"] = float(product_metrics_result.get("product_revenue", 0) or 0)
                    total_cost = float(product_metrics_result.get("total_cost", 0) or 0)
                    product["avg_price"] = float(product_metrics_result.get("avg_price", 0) or 0)
                    avg_cost = float(product_metrics_result.get("avg_cost", 0) or 0)
                    
                    # Calculate profitability metrics
                    units = product.get("units_sold", 0)
                    if units > 0:
                        product["revenue_per_unit"] = product["revenue"] / units
                        product["cost_per_unit"] = avg_cost
                        product["profit"] = product["revenue"] - total_cost
                        product["profit_per_unit"] = product["revenue_per_unit"] - avg_cost
                        # Profit margin as percentage
                        if product["revenue"] > 0:
                            product["profit_margin_pct"] = (product["profit"] / product["revenue"]) * 100
                        else:
                            product["profit_margin_pct"] = 0.0
                        # Contribution margin
                        product["contribution_margin"] = product["revenue_per_unit"] - avg_cost
                    else:
                        product["revenue_per_unit"] = 0.0
                        product["cost_per_unit"] = 0.0
                        product["profit"] = 0.0
                        product["profit_per_unit"] = 0.0
                        product["profit_margin_pct"] = 0.0
                        product["contribution_margin"] = 0.0
                else:
                    product["revenue"] = 0.0
                    product["avg_price"] = 0.0
                    product["revenue_per_unit"] = 0.0
                    product["cost_per_unit"] = 0.0
                    product["profit"] = 0.0
                    product["profit_per_unit"] = 0.0
                    product["profit_margin_pct"] = 0.0
                    product["contribution_margin"] = 0.0
            except Exception as e:
                logging.warning(f"Could not calculate metrics for product {product.get('sku')}: {e}")
                session.rollback()
                product["revenue"] = 0.0
                product["avg_price"] = 0.0
                product["revenue_per_unit"] = 0.0
                product["cost_per_unit"] = 0.0
                product["profit"] = 0.0
                product["profit_per_unit"] = 0.0
                product["profit_margin_pct"] = 0.0
                product["contribution_margin"] = 0.0
        
        logging.info("COLLECT SUPABASE SUCCESS", extra={
            "user_id": params.get("user_id"),
            "orders": orders_count,
            "revenue": revenue,
            "avg_order_value": avg_order_value,
            "top_products": [p["name"] for p in top_products],
            "period": period
        })
        
        return {
            "orders": orders_count,
            "revenue": float(revenue),
            "avg_order_value": avg_order_value,
            "top_products": top_products,
            "period": period,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }
    except Exception as e:
        # Log error and return empty structure for fail-safe behavior
        logging.error(f"Database query failed: {e}", exc_info=True)
        return {
            "orders": 0,
            "revenue": 0.0,
            "avg_order_value": 0.0,
            "top_products": [],
            "period": period,
            "start_date": "",
            "end_date": ""
        }
    finally:
        if session:
            session.close()

