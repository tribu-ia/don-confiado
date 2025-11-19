#!/usr/bin/env python3
"""
Script to populate Supabase with sample sales data for testing the report workflow.
This data aligns with Neo4j products and supports various business questions.
"""

import os
import sys
from datetime import datetime, timedelta
from random import randint, choice, uniform
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from business.common.connection import SessionLocal

load_dotenv()

# Products that align with Neo4j data
# Now includes cost data for profitability calculations
# Cost is typically 60-75% of selling price for retail
PRODUCTS = [
    {"sku": "LECHE-001", "nombre": "Leche Entera 1L", "precio": 3500.00, "costo": 2100.00},
    {"sku": "HUEVOS-001", "nombre": "Huevos AA x30", "precio": 12000.00, "costo": 7200.00},
    {"sku": "ARROZ-001", "nombre": "Arroz Diana 1kg", "precio": 4500.00, "costo": 2700.00},
    {"sku": "SALCH-001", "nombre": "Salchichas Zenu x6", "precio": 8500.00, "costo": 5100.00},
    {"sku": "BANANO-001", "nombre": "Banano x1kg", "precio": 2500.00, "costo": 1500.00},
    {"sku": "TOMATE-001", "nombre": "Tomate Chonto x1kg", "precio": 3200.00, "costo": 1920.00},
    {"sku": "CEBOLLA-001", "nombre": "Cebolla Larga x1kg", "precio": 2800.00, "costo": 1680.00},
    {"sku": "PAN-001", "nombre": "Pan Bimbo Integral", "precio": 5500.00, "costo": 3300.00},
    {"sku": "GALLETAS-001", "nombre": "Galletas Saltín Noel", "precio": 4200.00, "costo": 2520.00},
    {"sku": "CAFE-001", "nombre": "Café Colcafé 500g", "precio": 15000.00, "costo": 9000.00},
    {"sku": "ACEITE-001", "nombre": "Aceite de Cocina 1L", "precio": 8500.00, "costo": 5100.00},
    {"sku": "PASTA-001", "nombre": "Pasta Spaghetti 500g", "precio": 3200.00, "costo": 1920.00},
    {"sku": "ATUN-001", "nombre": "Atún en Lata 170g", "precio": 4500.00, "costo": 2700.00},
    {"sku": "COCA-001", "nombre": "Coca-Cola 1.5L", "precio": 4200.00, "costo": 2520.00},
    {"sku": "YOGURT-001", "nombre": "Yogurt Alpina 1L", "precio": 6500.00, "costo": 3900.00},
]

# Sample customer regions (aligned with Neo4j)
REGIONS = ["Bogotá D.C.", "Región Caribe", "Antioquia", "Región Pacífica", "Región Oriental"]


def ensure_tables_exist(session: Session):
    """Create tables if they don't exist"""
    print("Checking/creating tables...")
    
    # Create ventas table
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS ventas (
            id SERIAL PRIMARY KEY,
            cliente_id INTEGER,
            total NUMERIC(15, 2) NOT NULL,
            fecha_creacion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            region VARCHAR(100),
            estado VARCHAR(20) DEFAULT 'completada'
        )
    """))
    
    # Create venta_items table
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS venta_items (
            id SERIAL PRIMARY KEY,
            venta_id INTEGER NOT NULL,
            producto_sku VARCHAR(50) NOT NULL,
            producto_nombre VARCHAR(200) NOT NULL,
            cantidad INTEGER NOT NULL,
            precio_unitario NUMERIC(15, 2) NOT NULL,
            costo_unitario NUMERIC(15, 2) DEFAULT 0,
            subtotal NUMERIC(15, 2) NOT NULL,
            fecha_creacion TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (venta_id) REFERENCES ventas(id) ON DELETE CASCADE
        )
    """))
    
    # Add costo_unitario column if it doesn't exist (for existing tables)
    try:
        session.execute(text("""
            ALTER TABLE venta_items 
            ADD COLUMN IF NOT EXISTS costo_unitario NUMERIC(15, 2) DEFAULT 0
        """))
    except Exception:
        pass  # Column might already exist
    
    # Create index for better query performance
    session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_ventas_fecha ON ventas(fecha_creacion)
    """))
    
    session.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_venta_items_fecha ON venta_items(fecha_creacion)
    """))
    
    session.commit()
    print("✅ Tables created/verified")


def populate_products(session: Session):
    """Ensure products exist in productos table"""
    print("Populating products...")
    
    # Add costo_unitario column to productos if it doesn't exist
    try:
        session.execute(text("""
            ALTER TABLE productos 
            ADD COLUMN IF NOT EXISTS costo_unitario NUMERIC(15, 2) DEFAULT 0
        """))
    except Exception:
        pass  # Column might already exist
    
    for product in PRODUCTS:
        costo = product.get("costo", 0)
        session.execute(text("""
            INSERT INTO productos (sku, nombre, precio_venta, costo_unitario, cantidad, fecha_creacion)
            VALUES (:sku, :nombre, :precio, :costo, :cantidad, :fecha)
            ON CONFLICT (sku) DO UPDATE SET
                nombre = EXCLUDED.nombre,
                precio_venta = EXCLUDED.precio_venta,
                costo_unitario = EXCLUDED.costo_unitario,
                cantidad = productos.cantidad + EXCLUDED.cantidad
        """), {
            "sku": product["sku"],
            "nombre": product["nombre"],
            "precio": product["precio"],
            "costo": costo,
            "cantidad": randint(50, 500),
            "fecha": datetime.now()
        })
    
    session.commit()
    print(f"✅ {len(PRODUCTS)} products populated")


def generate_sales_data(session: Session, days_back: int = 90, sales_per_day: int = 15, fast_mode: bool = False):
    """Generate sample sales data for the last N days"""
    print(f"Generating sales data for last {days_back} days...")
    if fast_mode:
        print("  Using fast mode (bulk inserts)...")
    
    # Generate data up to today (not backwards from a past date)
    end_date = datetime.now().replace(hour=23, minute=59, second=59)
    start_date = (end_date - timedelta(days=days_back)).replace(hour=0, minute=0, second=0)
    
    sales_created = 0
    items_created = 0
    
    if fast_mode:
        # Fast mode: prepare all data, then bulk insert
        all_sales = []
        all_items = []
        
        current_date = start_date
        while current_date <= end_date:
            day_of_week = current_date.weekday()
            if day_of_week < 5:
                daily_sales = randint(sales_per_day - 5, sales_per_day + 5)
            else:
                daily_sales = randint(sales_per_day + 5, sales_per_day + 15)
            
            for _ in range(daily_sales):
                sale_time = current_date.replace(
                    hour=randint(8, 20),
                    minute=randint(0, 59),
                    second=randint(0, 59)
                )
                
                region = choice(REGIONS)
                total = 0.0
                sale_items = []
                
                num_items = randint(1, 5)
                for _ in range(num_items):
                    product = choice(PRODUCTS)
                    quantity = randint(1, 4)
                    subtotal = product["precio"] * quantity
                    total += subtotal
                    
                    sale_items.append({
                        "producto_sku": product["sku"],
                        "producto_nombre": product["nombre"],
                        "cantidad": quantity,
                        "precio_unitario": product["precio"],
                        "costo_unitario": product.get("costo", 0),
                        "subtotal": subtotal,
                        "fecha": sale_time
                    })
                
                all_sales.append({
                    "cliente_id": randint(1, 100),
                    "total": total,
                    "fecha": sale_time,
                    "region": region,
                    "estado": "completada",
                    "items": sale_items
                })
            
            current_date += timedelta(days=1)
        
        # Bulk insert sales
        print(f"  Inserting {len(all_sales)} sales in bulk...")
        for i, sale in enumerate(all_sales):
            result = session.execute(text("""
                INSERT INTO ventas (cliente_id, total, fecha_creacion, region, estado)
                VALUES (:cliente_id, :total, :fecha, :region, :estado)
                RETURNING id
            """), sale)
            venta_id = result.fetchone()[0]
            
            # Bulk insert items for this sale
            for item in sale["items"]:
                item["venta_id"] = venta_id
                all_items.append(item)
            
            if (i + 1) % 200 == 0:
                session.commit()
                print(f"    Progress: {i + 1}/{len(all_sales)} sales...")
        
        # Bulk insert all items
        print(f"  Inserting {len(all_items)} items in bulk...")
        for i, item in enumerate(all_items):
            session.execute(text("""
                INSERT INTO venta_items 
                (venta_id, producto_sku, producto_nombre, cantidad, precio_unitario, costo_unitario, subtotal, fecha_creacion)
                VALUES (:venta_id, :producto_sku, :producto_nombre, :cantidad, :precio_unitario, :costo_unitario, :subtotal, :fecha)
            """), item)
            
            if (i + 1) % 500 == 0:
                session.commit()
                print(f"    Progress: {i + 1}/{len(all_items)} items...")
        
        session.commit()
        sales_created = len(all_sales)
        items_created = len(all_items)
    else:
        # Original mode: insert one by one
        current_date = start_date
        while current_date <= end_date:
            day_of_week = current_date.weekday()
            if day_of_week < 5:
                daily_sales = randint(sales_per_day - 5, sales_per_day + 5)
            else:
                daily_sales = randint(sales_per_day + 5, sales_per_day + 15)
            
            for _ in range(daily_sales):
                sale_time = current_date.replace(
                    hour=randint(8, 20),
                    minute=randint(0, 59),
                    second=randint(0, 59)
                )
                
                region = choice(REGIONS)
                total = 0.0
                
                result = session.execute(text("""
                    INSERT INTO ventas (cliente_id, total, fecha_creacion, region, estado)
                    VALUES (:cliente_id, :total, :fecha, :region, :estado)
                    RETURNING id
                """), {
                    "cliente_id": randint(1, 100),
                    "total": 0.0,
                    "fecha": sale_time,
                    "region": region,
                    "estado": "completada"
                })
                
                venta_id = result.fetchone()[0]
                
                num_items = randint(1, 5)
                items = []
                for _ in range(num_items):
                    product = choice(PRODUCTS)
                    quantity = randint(1, 4)
                    price_unit = product["precio"]
                    cost_unit = product.get("costo", 0)
                    subtotal = price_unit * quantity
                    total += subtotal
                    
                    items.append({
                        "venta_id": venta_id,
                        "producto_sku": product["sku"],
                        "producto_nombre": product["nombre"],
                        "cantidad": quantity,
                        "precio_unitario": price_unit,
                        "costo_unitario": cost_unit,
                        "subtotal": subtotal,
                        "fecha": sale_time
                    })
                
                session.execute(text("""
                    UPDATE ventas SET total = :total WHERE id = :id
                """), {"total": total, "id": venta_id})
                
                for item in items:
                    session.execute(text("""
                        INSERT INTO venta_items
                        (venta_id, producto_sku, producto_nombre, cantidad, precio_unitario, costo_unitario, subtotal, fecha_creacion)
                        VALUES (:venta_id, :producto_sku, :producto_nombre, :cantidad, :precio_unitario, :costo_unitario, :subtotal, :fecha)
                    """), item)
                    items_created += 1
                
                sales_created += 1
            
            current_date += timedelta(days=1)
            
            if sales_created % 100 == 0:
                session.commit()
                print(f"  Progress: {sales_created} sales, {items_created} items...")
        
        session.commit()
    
    print(f"✅ Created {sales_created} sales with {items_created} items")


def verify_data(session: Session):
    """Verify the data was created correctly"""
    print("\nVerifying data...")
    
    # Check sales
    result = session.execute(text("""
        SELECT 
            COUNT(*) as total_ventas,
            SUM(total) as total_revenue,
            MIN(fecha_creacion) as primera_venta,
            MAX(fecha_creacion) as ultima_venta
        FROM ventas
    """)).fetchone()
    
    print(f"  Total ventas: {result[0]}")
    print(f"  Total revenue: ${result[1]:,.2f}")
    print(f"  Period: {result[2]} to {result[3]}")
    
    # Check top products
    result = session.execute(text("""
        SELECT 
            producto_nombre,
            SUM(cantidad) as total_unidades,
            SUM(subtotal) as total_revenue
        FROM venta_items
        GROUP BY producto_nombre
        ORDER BY total_unidades DESC
        LIMIT 5
    """)).fetchall()
    
    print("\n  Top 5 productos por unidades:")
    for row in result:
        print(f"    - {row[0]}: {row[1]} unidades, ${row[2]:,.2f}")
    
    # Check by region
    result = session.execute(text("""
        SELECT 
            region,
            COUNT(*) as ventas,
            SUM(total) as revenue
        FROM ventas
        GROUP BY region
        ORDER BY revenue DESC
    """)).fetchall()
    
    print("\n  Ventas por región:")
    for row in result:
        print(f"    - {row[0]}: {row[1]} ventas, ${row[2]:,.2f}")


def main():
    """Main execution"""
    print("="*70)
    print("POPULATING SUPABASE WITH SAMPLE SALES DATA")
    print("="*70)
    print()
    
    session: Session = None
    try:
        session = SessionLocal()
        
        # Step 1: Ensure tables exist
        ensure_tables_exist(session)
        print()
        
        # Step 2: Populate products
        populate_products(session)
        print()
        
        # Step 3: Generate sales data
        # Ask user for mode
        print("Select generation mode:")
        print("  1. Fast mode (30 days, ~450 sales) - ~30 seconds")
        print("  2. Standard mode (90 days, ~1,350 sales) - ~5-10 minutes")
        print("  3. Quick test (7 days, ~105 sales) - ~10 seconds")
        
        choice = input("\nEnter choice (1/2/3) [default: 1]: ").strip() or "1"
        
        if choice == "3":
            days = 7
            sales_per_day = 15
            fast = True
        elif choice == "2":
            days = 90
            sales_per_day = 15
            fast = False
        else:  # Default to fast
            days = 30
            sales_per_day = 15
            fast = True
        
        generate_sales_data(session, days_back=days, sales_per_day=sales_per_day, fast_mode=fast)
        print()
        
        # Step 4: Verify
        verify_data(session)
        print()
        
        print("="*70)
        print("✅ DATA POPULATION COMPLETE")
        print("="*70)
        print("\nThe database now has:")
        print("  - Sales data for the last 90 days")
        print("  - Product data aligned with Neo4j")
        print("  - Regional sales distribution")
        print("  - Support for various business questions")
        print("\nYou can now test queries like:")
        print("  - '¿Cuáles son los productos más vendidos?'")
        print("  - 'Genera un reporte de ventas del último mes'")
        print("  - '¿Qué productos tienen mejor desempeño?'")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        if session:
            session.rollback()
        sys.exit(1)
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    main()

