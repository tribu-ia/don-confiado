

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

productos = [
  {"nombre": "BARRA DE CEREAL TOSH FRESA x23", "precio": 1200},
  {"nombre": "BARRA DE CEREAL TOSH SABORES x6und", "precio": 7950},
  {"nombre": "CAFE COLCAFE 3 EN 1 SOBRE x19und", "precio": 700},
  {"nombre": "CAFE COLCAFE POLVO x250gr", "precio": 15990},
  {"nombre": "CAFE COLCAFE VAINILLA x50gr", "precio": 6400},
  {"nombre": "CAPUCHINO COLCAFE CLASICO x18sbr", "precio": 10550},
  {"nombre": "CAPUCHINO COLCAFE CLASICO x6sbr x18gr", "precio": 4850},
  {"nombre": "CHOCOLATE CHOCOLYNE BOLSA SPLENDA x120gr", "precio": 5500},
  {"nombre": "CHOCOLATE CHOCOLYNE CLAVOS Y CANELA BOLSA x120gr", "precio": 5900},
  {"nombre": "CHOCOLATE CHOCOLYNE SPLENDA BOLSA x120gr", "precio": 6400},
  {"nombre": "CHOCOLATE CHOCOLYNE SPLENDA CLASIC BARRA x156gr", "precio": 5100},
  {"nombre": "CHOCOLATE CORONA AZUCAR x250gr", "precio": 7950},
  {"nombre": "CHOCOLATE CORONA AZUCAR x500gr", "precio": 13950},
  {"nombre": "CHOCOLATE CORONA CLAVOS Y CANELA x250gr", "precio": 7950},
  {"nombre": "CHOCOLATINA GOL x31gr", "precio": 2500},
  {"nombre": "CHOCOLATINA JET BURBUJET VAINILLA", "precio": 5500},
  {"nombre": "CHOCOLATINA JET CREMA x18gr", "precio": 1190},
  {"nombre": "CHOCOLATINA JUMBO FLOW BLANCA x18gr", "precio": 1190},
  {"nombre": "CHOCOLATINA JUMBO FLOW BLANCA x48gr", "precio": 2500},
  {"nombre": "CHOCOLATINA JUMBO JET x100", "precio": 9000},
  {"nombre": "CHOCOLATINA JUMBO JET x40gr", "precio": 4800},
  {"nombre": "CHOCOLATINA JUMBO KREMANI x40gr", "precio": 1800},
  {"nombre": "CHOCOLATINA JUMBO MINI", "precio": 1500},
  {"nombre": "CHOCOLATINA JUMBO MIX", "precio": 5200},
  {"nombre": "CHOCOLATINA MONTBLANC x60gr", "precio": 4900},
  {"nombre": "CHOCOLISTO CROCANTE x100g", "precio": 3400},
  {"nombre": "CHOCOLISTO FAMIPACK x440gr", "precio": 10591},
  {"nombre": "GALLETA WAFER JET VAINILLA x22gr", "precio": 2600},
  {"nombre": "GALLETAS DUCALES PROVOCACION x6und", "precio": 6400},
  {"nombre": "GALLETAS DUCALES TENTACION TACO REF x160g", "precio": 3900},
  {"nombre": "GALLETAS DUCALES TENTACION TACO REF x160grs", "precio": 2950},
  {"nombre": "GALLETAS DUCALES x441g", "precio": 9300},
  {"nombre": "GALLETAS DUCALES x441grs", "precio": 8330},
  {"nombre": "GALLETAS DUCALES x9", "precio": 3450},
  {"nombre": "GALLETAS SALTIN 6 TACOS", "precio": 7650},
  {"nombre": "GALLETAS SALTIN INTEGRAL DOBLE FIBRA x9", "precio": 3450},
  {"nombre": "GALLETAS SALTIN QUESO MANTEQUILLA x9", "precio": 4300},
  {"nombre": "GALLETAS SALTIN SEMILLAS Y CAREALES 3 TACOS", "precio": 6600},
  {"nombre": "GALLETAS SALTIN TACO x110gr", "precio": 1500},
  {"nombre": "GALLETAS SALTIN TRIGO Y MAIZ 3 TACOS", "precio": 4350},
  {"nombre": "GALLETAS SALTIN x9", "precio": 3450},
  {"nombre": "MAIZ TOSTADO LA ESPECIAL x40gr", "precio": 1400},
  {"nombre": "MANI EL MANICERO x25gr", "precio": 1000},
  {"nombre": "MANI LA ESPECIAL ARANDANOS x45gr", "precio": 2100},
  {"nombre": "MANI LA ESPECIAL BARRA x38gr", "precio": 3000},
  {"nombre": "MANI LA ESPECIAL CON CHOCOLATES x44gr", "precio": 1900},
  {"nombre": "MANI LA ESPECIAL CON PASAS x50gr", "precio": 1600},
  {"nombre": "MANI LA ESPECIAL KRAKS x30gr", "precio": 900},
  {"nombre": "MANI LA ESPECIAL LIMON PIMIENTA x40gr", "precio": 3000},
  {"nombre": "MANI LA ESPECIAL MIX ALMENDRAS", "precio": 2200},
  {"nombre": "MANI LA ESPECIAL x50gr", "precio": 2975},
  {"nombre": "PASTA DORIA ESPAGUETTI AL HUEVO x250gr", "precio": 2750},
  {"nombre": "PASTA DORIA LASAGNA x200gr", "precio": 5250},
  {"nombre": "PASTA DORIA RANCHERO SPAGUETTI x250gr", "precio": 2750},
  {"nombre": "PASTA DORIA SPAGUETI POLLO x250gr", "precio": 2750},
  {"nombre": "PASTA DORIA SPAGUETI TOMATE x250gr", "precio": 2750},
  {"nombre": "PASTA DORIA SPAGUETTI MANTEQUILLA x220gr", "precio": 2750},
  {"nombre": "PASTA DORIA SPAGUETTI VERDURAS x250gr", "precio": 2750},
  {"nombre": "PASTA DORIA SPAGUETTI x500gr", "precio": 4200},
  {"nombre": "PASTA DORIA TORNILLOS VERDURAS x250gr", "precio": 2290},
    {"nombre": "BEBIDA GASIFICADA SABOR A VINO MOSTELO BLANCO x750ml", "precio": 13000},
  {"nombre": "BEBIDA SCHW X1,75LT SODA", "precio": 3500},
  {"nombre": "GASEOSA POSTOBON ACQUA FRUTOS ROJOS x1.5lt", "precio": 3000},
  {"nombre": "GASEOSA POSTOBON MANZANA x400ml", "precio": 2700},
  {"nombre": "GASEOSA 7UP SABOR A LIMA LIMON x2lt", "precio": 0},
  {"nombre": "GASEOSA BIG COLA LIMON x3.020lt", "precio": 3800},
  {"nombre": "GASEOSA BIG COLA x1700ml", "precio": 2100},
  {"nombre": "GASEOSA BIG COLA x3,020lt", "precio": 3750},
  {"nombre": "GASEOSA BIG COLA x400ml", "precio": 0},
  {"nombre": "GASEOSA BIG MANZANA x400ml", "precio": 0},
  {"nombre": "GASEOSA CANADA DRY X 2.5 LT", "precio": 5500},
  {"nombre": "GASEOSA CANADA DRY x400ml", "precio": 0},
  {"nombre": "GASEOSA COCA COLA CERO PLAST x250ml", "precio": 1785},
  {"nombre": "GASEOSA COCA COLA CON CAFE LATA x235ml", "precio": 0},
  {"nombre": "GASEOSA COCA COLA LIGHT x1,5ml", "precio": 5400},
  {"nombre": "GASEOSA COCA COLA OFERTA BT x1,5lt x2und", "precio": 6570},
  {"nombre": "GASEOSA COCA COLA PLAST x250ml", "precio": 2500},
  {"nombre": "GASEOSA COCA COLA PLAST x400ml", "precio": 4000},
  {"nombre": "GASEOSA COCA COLA SIN AZUCAR x400ml", "precio": 2500},
  {"nombre": "GASEOSA COCA COLA SIN CALORIAS SIN AZUCAR x600ml", "precio": 3400},
  {"nombre": "GASEOSA COCA COLA SIXPACK x235ml", "precio": 8750},
  {"nombre": "GASEOSA COCA COLA x1.5lts", "precio": 3500},
  {"nombre": "GASEOSA COCA COLA x1500ml", "precio": 5500},
  {"nombre": "GASEOSA COCA COLA x2000ml", "precio": 0},
  {"nombre": "GASEOSA COCA COLA x3lts", "precio": 9000},
  {"nombre": "GASEOSA COCA COLA ZERO x1.5ml", "precio": 7000},
  {"nombre": "GASEOSA COCA COLA ZERO x2,5lt", "precio": 5300},
  {"nombre": "GASEOSA COCA-COLA SIN AZÚCAR x330ml", "precio": 1400},
  {"nombre": "GASEOSA COLA ROMAN x1500ml", "precio": 4500},
  {"nombre": "GASEOSA FRIZZ FRUTOS ROJOS x1500ml", "precio": 3000},
  {"nombre": "GASEOSA GLACIAL MANZANA x1lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL MANZANA x400ml", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL COLA NEGRA x1.7lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL COLA NEGRA x1lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL CREMA SODA TIPO COLOMBIANA x1.7lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL CREMA SODA TIPO COLOMBIANA x1lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL CREMA SODA TIPO COLOMBIANA x400ml", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL KOLA ROJA TIPO PREMO x1.7lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL KOLA ROJA TIPO PREMO x1lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL KOLA ROJA TIPO PREMO x3,05 lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL LIMA LIMON x1.7lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL LIMA LIMON x3,05lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL LIMA LIMON x400ml", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL NARANJA x1.7lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL NARANJA x400ml", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL UVA x1lt", "precio": 999999},
  {"nombre": "GASEOSA GLACIAL UVA x3,05lt", "precio": 999999}
]


#------------------------------------------------------
# AQUI VAN LAS TOOLS
# Lo colocamos la anotación en tool en cada una
#---------------------------------------------------------
@tool
def buscar_productos_tool(texto_busqueda: str) -> list[dict]:
    """
    Puedes usar esta herramienta para buscar productos por nombre. Puedes usar nombres parciales.
    Para llamar la función debes mandar el nombre del producto en mayúsculas.

    Esta herramienta retornará el producto con su nombre, precios.

    La puedes usar si te piden precios de cierto tipos de productos

    """
    global  productos
    # Normaliza el texto y separa por palabras
    palabras = [p.lower() for p in texto_busqueda.strip().split() if p]

    # Filtra los productos que contienen todas las palabras
    resultados = []
    for prod in productos:
        nombre = prod.get("nombre", "").lower()
        if all(palabra in nombre for palabra in palabras):
            resultados.append(prod)
    return resultados

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
from langchain.tools import tool
from sqlalchemy.orm import Session
from sqlalchemy import or_
from business.entities.tercero import Tercero
from business.common.connection import SessionLocal




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