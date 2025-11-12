from langchain_core.tools import tool
from langchain.tools import tool
from sqlalchemy.orm import Session
from sqlalchemy import or_ , and_
from sqlalchemy.exc import IntegrityError, DataError
from typing import Optional
from business.entities.tercero import Tercero
from business.entities.producto import Producto
from business.dao.producto_dao import ProductoDAO
from business.dao.tercero_dao import TerceroDAO
from business.common.connection import SessionLocal



#------------------------------------------------------
# AQUI VAN LAS TOOLS
# Lo colocamos la anotación en tool en cada una
#---------------------------------------------------------
@tool
def buscar_productos_tool(texto_busqueda: str) -> list[dict]:
    """
    Busca productos por nombre (coincidencias parciales, sin importar mayúsculas/minúsculas).

    Puedes usar esta herramienta para obtener precios o información de productos específicos.
    Para llamarla, debes mandar el nombre del producto en mayúsculas (aunque no es obligatorio).

    Retorna una lista de productos con su nombre, precio, cantidad y proveedor.
    """
    session: Session = SessionLocal()
    try:
        producto_dao = ProductoDAO(session)
        
        # Normalizar y dividir palabras clave
        palabras = [p.strip().lower() for p in texto_busqueda.split() if p]
        condiciones = [Producto.nombre.ilike(f"%{palabra}%") for palabra in palabras]

        # Ejecutar la consulta usando la sesión del DAO
        query = (
            producto_dao.session.query(Producto)
            .filter(and_(*condiciones))
            .limit(20)
            .all()
        )

        # Convertir resultados a dict
        resultados = []
        for p in query:
            resultados.append({
                "id": p.id,
                "sku": p.sku,
                "nombre": p.nombre,
                "precio_venta": float(p.precio_venta),
                "cantidad": p.cantidad,
                "proveedor_id": p.proveedor_id,
                "proveedor": p.proveedor.razon_social if p.proveedor else None,
            })

        return resultados

    except Exception as e:
        return [{"error": str(e)}]
    finally:
        session.close()

@tool
def buscar_por_rango_de_precio(minimo: float, maximo: float) -> list[dict]:
    """
    Busca productos por rango de precio
    """

    session: Session = SessionLocal()
    try:
        producto_dao = ProductoDAO(session)
        query = (
            producto_dao.session.query(Producto)
            .filter(
                Producto.precio_venta >= minimo,
                Producto.precio_venta <= maximo,
            )
            .limit(50)
            .all()
        )

        resultados = []
        for p in query:
            resultados.append({
                "id": p.id,
                "sku": p.sku,
                "nombre": p.nombre,
                "precio_venta": float(p.precio_venta),
                "cantidad": p.cantidad,
                "proveedor_id": p.proveedor_id,
                "proveedor": p.proveedor.razon_social if p.proveedor else None,
            })

        return resultados
    except Exception as e:
        return [{"error": str(e)}]
    finally:
        session.close()


#---------------------------------------------------------
@tool
def buscar_terceros_tool(texto_busqueda: str,context: dict= None) -> list[dict]:
    """
    Busca terceros por nombre, razón social o número de documento.
    Retorna una lista de diccionarios con la información encontrada.
    """

    if context:
        print("Contexto recibido en buscar_terceros_tool:", context)
    else:
        print("No se recibió contexto en buscar_terceros_tool.")

    session: Session = SessionLocal()
    try:
        tercero_dao = TerceroDAO(session)
        query = (
            tercero_dao.session.query(Tercero)
            .filter(
                or_(
                    Tercero.nombres.ilike(f"%{texto_busqueda}%"),
                    Tercero.apellidos.ilike(f"%{texto_busqueda}%"),
                    Tercero.razon_social.ilike(f"%{texto_busqueda}%"),
                    Tercero.numero_documento.ilike(f"%{texto_busqueda}%"),
                )
            )
            .limit(20)
            .all()
        )

        resultados = []
        for t in query:
            resultados.append({
                "id": t.id,
                "tipo_documento": t.tipo_documento,
                "numero_documento": t.numero_documento,
                "razon_social": t.razon_social,
                "nombres": t.nombres,
                "apellidos": t.apellidos,
                "telefono_celular": t.telefono_celular,
                "tipo_tercero": t.tipo_tercero,
                "email": t.email,
            })

        return resultados

    except Exception as e:
        return [{"error": str(e)}]
    finally:
        session.close()

@tool
def crear_tercero_tool(
    tipo_documento: str,
    numero_documento: str,
    tipo_tercero: str,
    razon_social: Optional[str] = None,
    nombres: Optional[str] = None,
    apellidos: Optional[str] = None,
    telefono_fijo: Optional[str] = None,
    telefono_celular: Optional[str] = None,
    direccion: Optional[str] = None,
    email: Optional[str] = None,
    email_facturacion: Optional[str] = None,
) -> dict:
    """
    Crea un nuevo tercero en la base de datos.
    Campos obligatorios: tipo_documento ('CC','NIT','CE'), numero_documento, tipo_tercero ('cliente','proveedor','empleado').
    Retorna un diccionario con los datos del tercero creado o un error.
    """
    session: Session = SessionLocal()
    try:
        tercero_dao = TerceroDAO(session)
        nuevo = Tercero(
            tipo_documento=tipo_documento,
            numero_documento=numero_documento,
            tipo_tercero=tipo_tercero,
            razon_social=razon_social,
            nombres=nombres,
            apellidos=apellidos,
            telefono_fijo=telefono_fijo,
            telefono_celular=telefono_celular,
            direccion=direccion,
            email=email,
            email_facturacion=email_facturacion,
        )
        nuevo = tercero_dao.create(nuevo)
        return {
            "id": nuevo.id,
            "tipo_documento": nuevo.tipo_documento,
            "numero_documento": nuevo.numero_documento,
            "razon_social": nuevo.razon_social,
            "nombres": nuevo.nombres,
            "apellidos": nuevo.apellidos,
            "telefono_fijo": nuevo.telefono_fijo,
            "telefono_celular": nuevo.telefono_celular,
            "tipo_tercero": nuevo.tipo_tercero,
            "direccion": nuevo.direccion,
            "email": nuevo.email,
            "email_facturacion": nuevo.email_facturacion,
        }
    except (IntegrityError, DataError) as e:
        session.rollback()
        return {"error": "No fue posible crear el tercero", "detalle": str(e)}
    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()

@tool
def crear_producto_tool(
    sku: str,
    nombre: str,
    precio_venta: float,
    cantidad: int = 0,
    proveedor_id: Optional[int] = None,
) -> dict:
    """
    Crea un nuevo producto en la base de datos.
    Campos obligatorios: sku, nombre, precio_venta.
    Opcionales: cantidad (default 0), proveedor_id (FK a terceros.id).
    Retorna un diccionario con los datos del producto creado o un error.
    """
    session: Session = SessionLocal()
    try:
        producto_dao = ProductoDAO(session)
        tercero_dao = TerceroDAO(session)
        
        if proveedor_id is not None:
            proveedor = tercero_dao.findById(proveedor_id)
            if proveedor is None:
                return {"error": f"Proveedor con id {proveedor_id} no existe"}

        nuevo = Producto(
            sku=sku,
            nombre=nombre,
            precio_venta=precio_venta,
            cantidad=cantidad,
            proveedor_id=proveedor_id,
        )
        nuevo = producto_dao.create(nuevo)

        return {
            "id": nuevo.id,
            "sku": nuevo.sku,
            "nombre": nuevo.nombre,
            "precio_venta": float(nuevo.precio_venta),
            "cantidad": nuevo.cantidad,
            "proveedor_id": nuevo.proveedor_id,
            "proveedor": nuevo.proveedor.razon_social if nuevo.proveedor else None,
        }
    except (IntegrityError, DataError) as e:
        session.rollback()
        return {"error": "No fue posible crear el producto", "detalle": str(e)}
    except Exception as e:
        session.rollback()
        return {"error": str(e)}
    finally:
        session.close()


def create_tools_array():
    """Crea el array de tools para el agente del chatbot."""
    tools = [
        buscar_productos_tool,
        buscar_por_rango_de_precio,
        buscar_terceros_tool,
        crear_tercero_tool,
        crear_producto_tool,
    ]
    return tools  

##-----------------[ FIN DE LAS TOOLS] ------------------------