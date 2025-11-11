from langchain_core.tools import tool
from langchain.tools import tool
from sqlalchemy.orm import Session
from sqlalchemy import or_ , and_
from business.entities.tercero import Tercero
from business.entities.producto import Producto
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
        # Normalizar y dividir palabras clave
        palabras = [p.strip().lower() for p in texto_busqueda.split() if p]

        # Construir filtro dinámico para que todas las palabras coincidan
        condiciones = [Producto.nombre.ilike(f"%{palabra}%") for palabra in palabras]

        # Ejecutar la consulta
        query = session.query(Producto).filter(and_(*condiciones)).limit(20).all()

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

    global  productos
    resp = []
    for p in productos:
      if minimo <= p["precio"]  and p["precio"] <= maximo:
        resp.append(p)
    return resp


# --- NUEVA TOOL: Detectar despedidas ---

finalizar_chat = False

@tool
def detectar_despedida_tool(texto: str):
    """Debes invocar esta tool cuando el usuario se despida ."""
    global finalizar_chat
    finalizar_chat = True
    return True



#---------------------------------------------------------
@tool
def buscar_terceros_tool(texto_busqueda: str) -> list[dict]:
    """
    Busca terceros por nombre, razón social o número de documento.
    Retorna una lista de diccionarios con la información encontrada.
    """
    session: Session = SessionLocal()
    try:
        query = (
            session.query(Tercero)
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



def create_tools_array():
    """Crea el array de tools para el agente del chatbot."""
    tools = [
        buscar_productos_tool,
        buscar_por_rango_de_precio,
        detectar_despedida_tool,
        buscar_terceros_tool
    ]
    return tools  

##-----------------[ FIN DE LAS TOOLS] ------------------------