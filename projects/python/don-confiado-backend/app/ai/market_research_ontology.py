"""
Market Research Ontology Configuration
Specialized for family consumption market studies and product analysis
"""

def get_market_research_entities():
    """Get entities for market research and consumption analysis"""
    return [
        # Demographics
        "Persona", "Familia", "Hogar", "Consumidor", "Cliente", "Usuario",
        
        # Geographic
        "Pais", "Region", "Ciudad", "Zona", "Departamento", "Municipio",
        
        # Products and Services
        "Producto", "Servicio", "Categoria", "Marca", "Modelo", "Variante",
        "Alimento", "Bebida", "Electrodomestico", "Tecnologia", "Ropa",
        
        # Market Analysis
        "Mercado", "Segmento", "Nicho", "Industria", "Sector", "Cadena",
        "Distribuidor", "Retailer", "Proveedor", "Fabricante",
        
        # Consumption Patterns
        "Consumo", "Compra", "Venta", "Adquisicion", "Uso", "Frecuencia",
        "Cantidad", "Volumen", "Valor", "Precio", "Costo",
        
        # Time and Trends
        "Periodo", "Temporada", "Tendencia", "Crecimiento", "Declive",
        "Estacionalidad", "Ciclo", "Momento",
        
        # Economic Factors
        "Ingreso", "Gasto", "Presupuesto", "Renta", "Salario", "Poder_Adquisitivo",
        "Inflacion", "Economia", "Finanzas",
        
        # Behavioral
        "Comportamiento", "Preferencia", "Hábito", "Necesidad", "Motivacion",
        "Actitud", "Percepcion", "Satisfaccion",
        
        # Data and Research
        "Estudio", "Encuesta", "Dato", "Metrica", "Indicador", "KPI",
        "Analisis", "Reporte", "Hallazgo", "Conclusion"
    ]


def get_market_research_relations():
    """Get relationships for market research and consumption analysis"""
    return [
        # Geographic relationships
        "UBICADO_EN", "PERTENECE_A", "DIVIDIDO_EN", "CONTIENE",
        
        # Product relationships
        "PERTENECE_A_CATEGORIA", "ES_MARCA_DE", "COMPETIDOR_DE", "SUSTITUTO_DE",
        "COMPLEMENTARIO_DE", "VARIANTE_DE", "VERSION_DE",
        
        # Market relationships
        "OPERAR_EN", "COMPETIR_EN", "DOMINAR", "LIDERAR", "SEGMENTAR",
        "TARGET_A", "DIRIGIDO_A", "ENFOCADO_EN",
        
        # Consumption relationships
        "CONSUMIR", "COMPRAR", "VENDER", "ADQUIRIR", "USAR", "PREFERIR",
        "RECOMENDAR", "EVITAR", "REEMPLAZAR",
        
        # Economic relationships
        "GASTAR_EN", "INVERTIR_EN", "AHORRAR_PARA", "FINANCIAR",
        "COSTAR", "VALER", "PRECIAR_EN",
        
        # Behavioral relationships
        "INFLUIR_EN", "MOTIVAR", "SATISFACER", "FRUSTRAR", "ATRAER",
        "REPELER", "CONVENCER", "DISUADIR",
        
        # Temporal relationships
        "OCURRIR_EN", "DURAR", "CONTINUAR", "TERMINAR", "INICIAR",
        "ANTECEDER", "SUCEDER", "COINCIDIR_CON",
        
        # Data relationships
        "MEDIR", "ANALIZAR", "REPORTAR", "DOCUMENTAR", "REGISTRAR",
        "TRACKING", "MONITOREAR", "EVALUAR",
        
        # Family and demographic relationships
        "PERTENECER_A", "SER_MIEMBRO_DE", "VIVIR_EN", "TRABAJAR_EN",
        "EDUCAR_EN", "CRECER_EN", "ENVEJECER_EN",
        
        # Market dynamics
        "CRECER", "DECLINAR", "ESTABILIZAR", "FLUCTUAR", "PICAR",
        "RECUPERAR", "EXPANDIR", "CONTRATAR"
    ]


def get_market_research_extraction_prompt():
    """Get the extraction prompt for market research documents"""
    return """
You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Person", "properties": {{"name": "John"}} }}],
"relationships": [{{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {{"since": "2024-08-01"}} }}] }}

Use only the following node and relationship types (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
•⁠  ⁠Do not return any additional information other than the JSON in it.
•⁠  ⁠Omit any backticks around the JSON - simply output the JSON on its own.
•⁠  ⁠The JSON object must not wrapped into a list - it is its own JSON object.
•⁠  ⁠Property names must be enclosed in double quotes

Examples:
{examples}

Input text:

{text}
"""
