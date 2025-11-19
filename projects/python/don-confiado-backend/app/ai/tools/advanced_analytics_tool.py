"""
Advanced Analytics Tool for generating deeper business insights.
Provides trend analysis, regional comparisons, time patterns, and more.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text
from langchain_core.tools import tool
from datetime import datetime, timedelta
import logging
from business.common.connection import SessionLocal

load_dotenv()


def _parse_period(period: str) -> Tuple[datetime, datetime]:
    """Parse period string to date range."""
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
        days = 30
    
    start_date = end_date - timedelta(days=days)
    return start_date, end_date


@tool
def advanced_analytics_tool(params: dict) -> dict:
    """
    Advanced analytics tool providing trend analysis, regional insights, 
    time patterns, and comparative metrics.
    
    Args:
        params: Dict with query parameters:
            - period: "last_30_days", "last_7_days", etc.
            - analysis_type: "trends", "regional", "time_patterns", "comparison", "all"
    
    Returns:
        Dict with advanced analytics:
            - trends: Daily/weekly trends
            - regional_performance: Regional breakdown
            - time_patterns: Day of week, growth rates
            - period_comparison: Current vs previous period
            - insights: Key findings
    """
    period = params.get("period", "last_30_days")
    analysis_type = params.get("analysis_type", "all")
    session: Optional[Session] = None
    
    try:
        start_date, end_date = _parse_period(period)
        previous_start = start_date - (end_date - start_date)
        previous_end = start_date
        
        session = SessionLocal()
        results = {
            "period": period,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
        
        # 1. TREND ANALYSIS - Daily trends
        if analysis_type in ["trends", "all"]:
            try:
                daily_trends = session.execute(text("""
                    SELECT 
                        DATE(fecha_creacion) as date,
                        COUNT(*) as orders,
                        SUM(total) as revenue,
                        AVG(total) as avg_order_value
                    FROM ventas
                    WHERE DATE(fecha_creacion) >= DATE(:start_date)
                      AND DATE(fecha_creacion) <= DATE(:end_date)
                    GROUP BY DATE(fecha_creacion)
                    ORDER BY date DESC
                """), {
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().all()
                
                if daily_trends:
                    results["daily_trends"] = [
                        {
                            "date": str(row["date"]),
                            "orders": int(row["orders"]),
                            "revenue": float(row["revenue"] or 0),
                            "avg_order_value": float(row["avg_order_value"] or 0)
                        }
                        for row in daily_trends
                    ]
                    
                    # Calculate growth rate (last 7 days vs previous 7 days)
                    if len(daily_trends) >= 7:
                        recent_7 = sum(d["revenue"] for d in results["daily_trends"][:7])
                        previous_7 = sum(d["revenue"] for d in results["daily_trends"][7:14]) if len(daily_trends) >= 14 else recent_7
                        if previous_7 > 0:
                            growth_rate = ((recent_7 - previous_7) / previous_7) * 100
                            results["weekly_growth_rate"] = round(growth_rate, 2)
            except Exception as e:
                logging.warning(f"Trend analysis failed: {e}")
                session.rollback()
        
        # 2. REGIONAL ANALYSIS
        if analysis_type in ["regional", "all"]:
            try:
                regional_data = session.execute(text("""
                    SELECT 
                        region,
                        COUNT(*) as orders,
                        SUM(total) as revenue,
                        AVG(total) as avg_order_value,
                        SUM(total) / NULLIF(COUNT(*), 0) as revenue_per_order
                    FROM ventas
                    WHERE DATE(fecha_creacion) >= DATE(:start_date)
                      AND DATE(fecha_creacion) <= DATE(:end_date)
                      AND region IS NOT NULL
                    GROUP BY region
                    ORDER BY revenue DESC
                """), {
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().all()
                
                if regional_data:
                    results["regional_performance"] = [
                        {
                            "region": row["region"],
                            "orders": int(row["orders"]),
                            "revenue": float(row["revenue"] or 0),
                            "avg_order_value": float(row["avg_order_value"] or 0),
                            "revenue_per_order": float(row["revenue_per_order"] or 0)
                        }
                        for row in regional_data
                    ]
                    
                    # Calculate regional market share
                    total_revenue = sum(r["revenue"] for r in results["regional_performance"])
                    if total_revenue > 0:
                        for region in results["regional_performance"]:
                            region["market_share_pct"] = round((region["revenue"] / total_revenue) * 100, 2)
            except Exception as e:
                logging.warning(f"Regional analysis failed: {e}")
                session.rollback()
        
        # 3. TIME PATTERNS - Day of week analysis
        if analysis_type in ["time_patterns", "all"]:
            try:
                day_of_week = session.execute(text("""
                    SELECT 
                        EXTRACT(DOW FROM fecha_creacion) as day_of_week,
                        CASE EXTRACT(DOW FROM fecha_creacion)
                            WHEN 0 THEN 'Domingo'
                            WHEN 1 THEN 'Lunes'
                            WHEN 2 THEN 'Martes'
                            WHEN 3 THEN 'Miércoles'
                            WHEN 4 THEN 'Jueves'
                            WHEN 5 THEN 'Viernes'
                            WHEN 6 THEN 'Sábado'
                        END as day_name,
                        COUNT(*) as orders,
                        SUM(total) as revenue,
                        AVG(total) as avg_order_value
                    FROM ventas
                    WHERE DATE(fecha_creacion) >= DATE(:start_date)
                      AND DATE(fecha_creacion) <= DATE(:end_date)
                    GROUP BY EXTRACT(DOW FROM fecha_creacion)
                    ORDER BY day_of_week
                """), {
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().all()
                
                if day_of_week:
                    results["day_of_week_patterns"] = [
                        {
                            "day_name": row["day_name"],
                            "day_of_week": int(row["day_of_week"]),
                            "orders": int(row["orders"]),
                            "revenue": float(row["revenue"] or 0),
                            "avg_order_value": float(row["avg_order_value"] or 0)
                        }
                        for row in day_of_week
                    ]
            except Exception as e:
                logging.warning(f"Time pattern analysis failed: {e}")
                session.rollback()
        
        # 4. PERIOD COMPARISON - Current vs Previous
        if analysis_type in ["comparison", "all"]:
            try:
                # Current period
                current = session.execute(text("""
                    SELECT 
                        COUNT(*) as orders,
                        SUM(total) as revenue,
                        AVG(total) as avg_order_value
                    FROM ventas
                    WHERE DATE(fecha_creacion) >= DATE(:start_date)
                      AND DATE(fecha_creacion) <= DATE(:end_date)
                """), {
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().first()
                
                # Previous period
                previous = session.execute(text("""
                    SELECT 
                        COUNT(*) as orders,
                        SUM(total) as revenue,
                        AVG(total) as avg_order_value
                    FROM ventas
                    WHERE DATE(fecha_creacion) >= DATE(:prev_start)
                      AND DATE(fecha_creacion) < DATE(:prev_end)
                """), {
                    "prev_start": previous_start,
                    "prev_end": previous_end
                }).mappings().first()
                
                if current and previous:
                    current_orders = int(current["orders"] or 0)
                    current_revenue = float(current["revenue"] or 0)
                    prev_orders = int(previous["orders"] or 0)
                    prev_revenue = float(previous["revenue"] or 0)
                    
                    results["period_comparison"] = {
                        "current": {
                            "orders": current_orders,
                            "revenue": current_revenue,
                            "avg_order_value": float(current["avg_order_value"] or 0)
                        },
                        "previous": {
                            "orders": prev_orders,
                            "revenue": prev_revenue,
                            "avg_order_value": float(previous["avg_order_value"] or 0)
                        },
                        "changes": {
                            "orders_change_pct": round(((current_orders - prev_orders) / prev_orders * 100) if prev_orders > 0 else 0, 2),
                            "revenue_change_pct": round(((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0, 2)
                        }
                    }
            except Exception as e:
                logging.warning(f"Period comparison failed: {e}")
                session.rollback()
        
        # 5. PRODUCT MIX ANALYSIS - Top product combinations
        if analysis_type in ["product_mix", "all"]:
            try:
                # Products frequently sold together (same order)
                product_combinations = session.execute(text("""
                    SELECT 
                        vi1.producto_nombre as product1,
                        vi2.producto_nombre as product2,
                        COUNT(DISTINCT vi1.venta_id) as co_occurrence
                    FROM venta_items vi1
                    INNER JOIN venta_items vi2 ON vi1.venta_id = vi2.venta_id
                    WHERE vi1.producto_sku < vi2.producto_sku
                      AND DATE(vi1.fecha_creacion) >= DATE(:start_date)
                      AND DATE(vi1.fecha_creacion) <= DATE(:end_date)
                    GROUP BY vi1.producto_nombre, vi2.producto_nombre
                    HAVING COUNT(DISTINCT vi1.venta_id) >= 3
                    ORDER BY co_occurrence DESC
                    LIMIT 10
                """), {
                    "start_date": start_date,
                    "end_date": end_date
                }).mappings().all()
                
                if product_combinations:
                    results["product_combinations"] = [
                        {
                            "product1": row["product1"],
                            "product2": row["product2"],
                            "co_occurrence": int(row["co_occurrence"])
                        }
                        for row in product_combinations
                    ]
            except Exception as e:
                logging.warning(f"Product mix analysis failed: {e}")
                session.rollback()
        
        # 6. KEY INSIGHTS SUMMARY
        insights = []
        
        if "period_comparison" in results:
            comp = results["period_comparison"]
            if comp["changes"]["revenue_change_pct"] > 0:
                insights.append(f"Crecimiento de ingresos: {comp['changes']['revenue_change_pct']:.1f}% vs período anterior")
            elif comp["changes"]["revenue_change_pct"] < 0:
                insights.append(f"Decrecimiento de ingresos: {comp['changes']['revenue_change_pct']:.1f}% vs período anterior")
        
        if "regional_performance" in results and results["regional_performance"]:
            top_region = results["regional_performance"][0]
            insights.append(f"Región líder: {top_region['region']} con {top_region['market_share_pct']:.1f}% del mercado")
        
        if "day_of_week_patterns" in results and results["day_of_week_patterns"]:
            best_day = max(results["day_of_week_patterns"], key=lambda x: x["revenue"])
            insights.append(f"Día más fuerte: {best_day['day_name']} con ${best_day['revenue']:,.0f} en ingresos")
        
        if "weekly_growth_rate" in results:
            if results["weekly_growth_rate"] > 0:
                insights.append(f"Tendencia positiva: {results['weekly_growth_rate']:.1f}% de crecimiento semanal")
        
        results["insights"] = insights
        
        logging.info("ADVANCED ANALYTICS SUCCESS", extra={
            "user_id": params.get("user_id"),
            "analysis_type": analysis_type,
            "insights_count": len(insights)
        })
        
        return results
        
    except Exception as e:
        logging.error(f"Advanced analytics failed: {e}", exc_info=True)
        if session:
            session.rollback()
        return {
            "period": period,
            "error": str(e),
            "insights": []
        }
    finally:
        if session:
            session.close()

