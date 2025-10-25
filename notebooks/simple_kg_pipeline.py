# %% [markdown]
# # SimpleKGPipeline Script
# 
# This script contains the minimal code required to run the SimpleKGPipeline fragment.
# All imports are at the top as requested.

# %% [markdown]
# ## Cell 1: All required imports

# %%
# All required imports
import os
import asyncio
import neo4j
from neo4j_graphrag.llm import OpenAILLM as LLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter

# %% [markdown]
# ## Cell 2: Environment setup and Neo4j connection

# %%
# Environment setup and Neo4j connection
# Set up environment variables
os.environ["NEO4J_URI"] = "neo4j+s://6ec3c173.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "pUF62lvAt8ESNINGSmqpFabukdHLV6emQbrKA_V46HI"
os.environ["NEO4J_DATABASE"] = "neo4j"

# Neo4j connection
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_URI = os.environ["NEO4J_URI"]

neo_auth = (NEO4J_USERNAME, NEO4J_PASSWORD)
neo4j_driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=neo_auth)

# %% [markdown]
# ## Cell 3: Clean the database before inserting data

# %%
# Clean the database before inserting data
delete_query = """
MATCH (n)
DETACH DELETE n
"""
print("Cleaning database...")
neo4j_driver.execute_query(delete_query)
print("Database cleaned successfully!")

# %% [markdown]
# ## Cell 4: Initialize LLM and embedder

# %%
# Initialize LLM and embedder
ex_llm = LLM(
    model_name="gpt-4o-mini",
    model_params={
        "response_format": {"type": "json_object"},
        "temperature": 0
    }
)

embedder = Embeddings()

# %% [markdown]
# ## Cell 5: Define entities and relationships for Don Quixote

# %%
# Define entities and relationships for Don Quixote story
# Relationship types (directed)
rel_types = [
    "LIVES_IN",
    "OWNS", 
    "SERVES",
    "KNOWS",
    "ENCOUNTERS",
    "TRAVELS_TO",
    "PROTECTS",
    "FIGHTS",
    "LOVES",
    "WORKS_FOR",
    "BELONGS_TO",
    "MEETS",
    "HELPS",
    "ATTACKS",
    "DEFENDS"
]

# Entity types for Don Quixote story
basic_node_labels = [
    "Person", 
    "Character", 
    "Place", 
    "Object", 
    "Animal", 
    "Building", 
    "Weapon", 
    "Book", 
    "Adventure", 
    "Event", 
    "Profession", 
    "Title",
    "Location",
    "Castle",
    "Village"
]

node_labels = basic_node_labels

# Create a specialized prompt template for Don Quixote story
llm_graph_instruction = """
You are an expert at analyzing literary texts and extracting knowledge graphs. 
Focus on identifying:

ENTITIES:
- Characters (people, animals, fictional beings)
- Places (villages, castles, locations, geographical areas)
- Objects (weapons, books, items, possessions)
- Buildings (castles, inns, houses, structures)
- Events (adventures, battles, encounters, ceremonies)
- Professions (knights, merchants, farmers, innkeepers)
- Titles and roles (knight, squire, damsel, etc.)

RELATIONSHIPS:
- Physical relationships (lives_in, owns, travels_to)
- Social relationships (serves, knows, meets, loves)
- Action relationships (fights, protects, helps, attacks, defends)
- Professional relationships (works_for, belongs_to)
- Narrative relationships (encounters, participates_in)

Extract entities and relationships from the following text about Don Quixote's adventures.
Focus on the main characters, their relationships, the places they visit, and the events that occur.

Return the result as JSON using this format:
{{
  "nodes": [
    {{"id": "0", "label": "Person", "properties": {{"name": "Don Quixote", "description": "Main character"}}}}
  ],
  "relationships": [
    {{"type": "OWNS", "start_node_id": "0", "end_node_id": "1", "properties": {{"description": "Don Quixote owns his horse"}}}}
  ]
}}

Input text:

{{text}}
"""

# %% [markdown]
# ## Cell 6: Create SimpleKGPipeline (the exact fragment from the code)

# %%
# Create SimpleKGPipeline (the exact fragment from the code)
kg_builder_pdf = SimpleKGPipeline(
    llm=ex_llm,
    driver=neo4j_driver,
    text_splitter=FixedSizeSplitter(chunk_size=1500, chunk_overlap=200),
    embedder=embedder,
    entities=node_labels,
    relations=rel_types,
    prompt_template=llm_graph_instruction,
    from_pdf=False,
    perform_entity_resolution=True
)

# %% [markdown]
# ## Cell 7: Load Don Quixote text and run the pipeline

# %%
# Load Don Quixote text from file
with open("/Volumes/Life-OS/Users/Arkatechie/Development/tribu/don-confiado/notebooks/don-quijote-cap3.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Loaded text with {len(text)} characters")
print(f"First 200 characters: {text[:200]}...")

# Run the pipeline
print("Running SimpleKGPipeline on Don Quixote text...")
await kg_builder_pdf.run_async(text=text)
print("Pipeline execution completed!")

# %% [markdown]
# ## Cell 8: Create vector index for retrieval

# %%
# Create vector index for retrieval
from neo4j_graphrag.indexes import create_vector_index

print("Creating vector index...")
create_vector_index(
    neo4j_driver, 
    name="text_embeddings", 
    label="Chunk",
    embedding_property="embedding", 
    dimensions=1536, 
    similarity_fn="cosine"
)
print("✅ Vector index created successfully!")

# %% [markdown]
# ## Cell 9: Vector Retriever - Setup and Test

# %%
# Test queries for vector retrieval
test_queries = [
    "Who is Don Quixote?",
    "What is Rocinante?",
    "Tell me about the inn and the innkeeper",
    "What adventures did Don Quixote have?"
]

# %%
# Import and setup Vector Retriever
from neo4j_graphrag.retrievers import VectorRetriever
import json

# Initialize Vector Retriever
print("Initializing Vector Retriever...")

vector_retriever = VectorRetriever(
    neo4j_driver,
    index_name="text_embeddings",
    embedder=embedder
)
print("✅ VectorRetriever initialized successfully!")

print("\nTesting Vector Retriever...")
print("=" * 60)

for i, query in enumerate(test_queries, 1):
    print(f"\n🔍 QUERY {i}: {query}")
    print("-" * 50)
    
    # Vector Retriever
    print("\n📊 VECTOR RETRIEVER RESULTS:")
    try:
        vector_resp = vector_retriever.get_search_results(query_text=query, top_k=3)
        for j, record in enumerate(vector_resp.records, 1):
            print(f"  {j}. {json.dumps(record, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)

print("\n✅ Vector retrieval testing completed!")

# %% [markdown]
# ## Cell 10: Vector Cypher Retriever - Setup and Test

import neo4j
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
from neo4j_graphrag.retrievers import VectorRetriever,HybridRetriever,VectorCypherRetriever

from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.llm import OpenAILLM as LLM
from langchain.embeddings.openai import OpenAIEmbeddings
import json
import os



retrieval_query = """MATCH (n)-[:FROM_CHUNK]->(node)
RETURN collect(n) AS sources
"""

retrieval_query = """


WITH node AS chunk

// Buscar nodos _Entity_ conectados a este chunk
OPTIONAL MATCH (e1:_Entity_)-[:FROM_CHUNK]->(chunk)

// Obtener relaciones entre entidades
OPTIONAL MATCH (e1)-[r]-(e2:_Entity_)
WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK']

// Obtener los chunks de origen de las entidades relacionadas
OPTIONAL MATCH (e2)-[:FROM_CHUNK]->(c2:Chunk)

RETURN
    chunk,
    collect(DISTINCT e1) AS entities_1,
    collect(DISTINCT e2) AS entities_2,
    collect(DISTINCT r) AS relationships,
    collect(DISTINCT c2) AS related_chunks


"""


retrieval_query = """
WITH node AS chunk

// Buscar nodos _Entity_ conectados a este chunk
OPTIONAL MATCH (e1:_Entity_)-[:FROM_CHUNK]->(chunk)

// Obtener relaciones entre entidades
OPTIONAL MATCH (e1)-[r]-(e2:_Entity_)
WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK']

// Obtener los chunks de origen de las entidades relacionadas
OPTIONAL MATCH (e2)-[:FROM_CHUNK]->(c2:Chunk)

// Retornar sin incluir la propiedad 'embedding'
RETURN
    chunk { .* , embedding: null } AS chunk,
    [e IN collect(DISTINCT e1) | e { .* , embedding: null }] AS entities_1,
    [e IN collect(DISTINCT e2) | e { .* , embedding: null }] AS entities_2,
    collect(DISTINCT r) AS relationships,
    [c IN collect(DISTINCT c2) | c { .* , embedding: null }] AS related_chunks


"""

retrieval_query = """
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)-[r]->(m)
RETURN chunk, r, m, similarity_score
LIMIT 5
"""

retrieval_query = "MATCH (node) RETURN node"

retrieval_query_c2 = """
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)-[r]->(child)-[rr]->(child2)
RETURN 
  similarity_score, 
  r,
  rr,
  apoc.map.removeKey(properties(chunk), 'embedding') AS chunk,
  apoc.map.removeKey(properties(child), 'embedding') AS child,
  apoc.map.removeKey(properties(child2), 'embedding') AS child2
ORDER BY similarity_score DESC
"""


retrieval_query_u = """
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)
RETURN DISTINCT 
  similarity_score, 
  apoc.map.removeKey(properties(chunk), 'embedding') AS chunk
UNION 
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)-[r]->(child)-[rr]->(child2)
RETURN DISTINCT 
  similarity_score, 
  apoc.map.removeKey(properties(child), 'embedding') AS chunk
ORDER BY similarity_score DESC

"""

re_hops ="""
//1) Go out up to N hops in the entity graph and get relationships
WITH node AS chunk
MATCH (chunk)<-[:FROM_CHUNK]-()-[relList:!FROM_CHUNK]-{1,4}()
UNWIND relList AS rel

//2) collect relationships and text chunks
WITH collect(DISTINCT chunk) AS chunks,
 collect(DISTINCT rel) AS rels

//3) format and return context
RETURN '=== text ===n' + apoc.text.join([c in chunks | c.text], 'n---n') + 'nn=== kg_rels ===n' +
 apoc.text.join([r in rels | startNode(r).name + ' - ' + type(r) + '(' + coalesce(r.details, '') + ')' +  ' -> ' + endNode(r).name ], 'n---n') AS info
"""
pregunta = "en que club juega el arquero a quien le sacaron tarjeta amarilla por juego peligroso"

vector_cypher_retriever = VectorCypherRetriever(neo4j_driver, "text_embeddings", re_hops, embedder)
vc_resp = vector_cypher_retriever.get_search_results(query_text=pregunta, top_k=5)
pregunta = "en que club juega el arquero a quien le sacaron tarjeta amarilla por juego peligroso"

print("VectorCypherRetriever ",pregunta)
print(vc_resp)
print("------------")
for record  in vc_resp.records:
  print(type(record))
  print(record)
  #print("===========\n" + json.dumps(record, indent=4))

print("====================================================")


# %% [markdown]
# ## Cell 11: GraphRAG + LLM Multimodal Integration
# 
# This example shows how to combine GraphRAG retrieval results with Gemini LLM for enhanced question answering.
# We'll use both Vector Retriever and Vector Cypher Retriever results and send them to Gemini for processing.

# %%
# Import Langchain components for Gemini integration
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Initialize Gemini model using Langchain
print("Initializing Gemini model for enhanced retrieval processing...")
gemini_model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
print("✅ Gemini model initialized!")

# %%
# Create enhanced retrieval function that combines both retrievers
def enhanced_retrieval(query_text: str, top_k: int = 3):
    """
    Enhanced retrieval that combines Vector Retriever and Vector Cypher Retriever results
    """
    print(f"\n🔍 Enhanced Retrieval for: '{query_text}'")
    print("=" * 60)
    
    # Get Vector Retriever results
    print("\n📊 VECTOR RETRIEVER RESULTS:")
    try:
        vector_resp = vector_retriever.get_search_results(query_text=query_text, top_k=top_k)
        vector_results = []
        for i, record in enumerate(vector_resp.records, 1):
            print(f"  {i}. Score: {record.get('score', 'N/A')}")
            if 'text' in record:
                print(f"     Text: {record['text'][:200]}...")
            vector_results.append(record)
    except Exception as e:
        print(f"  Error: {e}")
        vector_results = []
    
    # Get Vector Cypher Retriever results
    print("\n🔗 VECTOR CYPHER RETRIEVER RESULTS:")
    try:
        vc_resp = vector_cypher_retriever.get_search_results(query_text=query_text, top_k=top_k)
        cypher_results = []
        print(vc_resp)
        if hasattr(vc_resp, 'records'):
            for i, record in enumerate(vc_resp.records, 1):
                print(f"  Record {i}:")
                chunk = record.get('chunk')
                if chunk and hasattr(chunk, '__dict__'):
                    chunk_dict = dict(chunk)
                    if 'text' in chunk_dict:
                        print(f"    Chunk text: {chunk_dict['text'][:150]}...")
                cypher_results.append(record)
        else:
            print(f"  Response: {str(vc_resp)}")
    except Exception as e:
        print(f"  Error: {e}")
        cypher_results = []
    
    return {
        "vector_results": vector_results,
        "cypher_results": cypher_results,
        "query": query_text
    }

# %%
# Create function to process retrieval results with Gemini - SINGLE METHOD VERSION
def process_with_gemini(retrieval_results: list, user_question: str, method_type: str = "vector"):
    """
    Process retrieval results with Gemini LLM for enhanced question answering.
    This version processes either vector OR cypher results (not both).
    
    Args:
        retrieval_results: List of results from either vector or cypher retrieval
        user_question: The user's question
        method_type: Either "vector" or "cypher" to indicate the retrieval method
    """
    # Prepare context based on method type
    if method_type == "vector":
        print("\n" + "="*80)
        print("🔍 ESTRUCTURA DETALLADA DE RESULTADOS VECTORIALES")
        print("="*80)
        print(f"Tipo de método: {method_type}")
        print(f"Número total de resultados: {len(retrieval_results)}")
        print("\nEstructura de cada resultado:")
        for i, result in enumerate(retrieval_results, 1):
            print(f"\n--- Resultado {i} ---")
            print(f"Tipo: {type(result)}")
            if isinstance(result, dict):
                print(f"Claves disponibles: {list(result.keys())}")
                for key, value in result.items():
                    print(f"  {key}: {type(value)} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            else:
                print(f"Valor: {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
        print("="*80)
        
        context_text = "RESULTADOS DE BÚSQUEDA VECTORIAL (SIN RELACIONES DE GRAFO):\n"
        context_text += "=" * 60 + "\n"
        for i, result in enumerate(retrieval_results, 1):
            if isinstance(result, dict) and 'text' in result:
                context_text += f"Resultado {i} (Score: {result.get('score', 'N/A')}):\n"
                context_text += f"{result['text']}\n\n"
    else:  # cypher
        print("\n" + "="*80)
        print("🔍 ESTRUCTURA DETALLADA DE RESULTADOS CYPHER")
        print("="*80)
        print(f"Tipo de método: {method_type}")
        print(f"Número total de resultados: {len(retrieval_results)}")
        print("\nEstructura de cada resultado:")
        for i, result in enumerate(retrieval_results, 1):
            print(f"\n--- Resultado {i} ---")
            print(f"Tipo: {type(result)}")
            if isinstance(result, dict):
                print(f"Claves disponibles: {list(result.keys())}")
                for key, value in result.items():
                    if key == 'chunk' and hasattr(value, '__dict__'):
                        print(f"  {key}: {type(value)} (objeto con atributos)")
                        chunk_dict = dict(value)
                        print(f"    Atributos del chunk: {list(chunk_dict.keys())}")
                        for attr_key, attr_value in chunk_dict.items():
                            print(f"      {attr_key}: {type(attr_value)} = {str(attr_value)[:100]}{'...' if len(str(attr_value)) > 100 else ''}")
                    elif key == 'chunks':
                        print(f"  {key}: {type(value)} (array de chunks)")
                        print(f"    Cantidad: {len(value) if isinstance(value, list) else 'N/A'}")
                        if isinstance(value, list) and value:
                            print(f"    Primer chunk: {type(value[0])}")
                    elif key == 'rels':
                        print(f"  {key}: {type(value)} (array de relaciones)")
                        print(f"    Cantidad: {len(value) if isinstance(value, list) else 'N/A'}")
                        if isinstance(value, list) and value:
                            print(f"    Primera relación: {type(value[0])}")
                    elif key == 'info':
                        print(f"  {key}: {type(value)} (formato multi-hop GraphRAG)")
                        print(f"    Contenido: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
                    else:
                        print(f"  {key}: {type(value)} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
            else:
                print(f"Valor: {str(result)[:100]}{'...' if len(str(result)) > 100 else ''}")
        print("="*80)
        
        context_text = "RESULTADOS DE BÚSQUEDA CON RELACIONES DE GRAFO (CYPHER):\n"
        context_text += "=" * 60 + "\n"
        for i, result in enumerate(retrieval_results, 1):
            if isinstance(result, dict):
                # Check if this is the new hopsCypherQuery format (returns 'chunks' and 'rels' fields)
                if 'chunks' in result and 'rels' in result:
                    context_text += f"Resultado {i} (Multi-hop GraphRAG):\n"
                    context_text += "=== TEXT CHUNKS ===\n"
                    for j, chunk in enumerate(result['chunks'], 1):
                        if hasattr(chunk, 'text'):
                            context_text += f"Chunk {j}: {chunk.text}\n"
                        elif isinstance(chunk, dict) and 'text' in chunk:
                            context_text += f"Chunk {j}: {chunk['text']}\n"
                    context_text += "\n=== KNOWLEDGE GRAPH RELATIONSHIPS ===\n"
                    for j, rel in enumerate(result['rels'], 1):
                        context_text += f"Relationship {j}: {type(rel)}\n"
                    context_text += "\n"
                # Check for old 'info' format
                elif 'info' in result:
                    context_text += f"Resultado {i} (Multi-hop GraphRAG):\n"
                    context_text += f"{result['info']}\n\n"
                # Fallback to old format for backward compatibility
                elif 'chunk' in result:
                    chunk = result.get('chunk')
                    if chunk and hasattr(chunk, '__dict__'):
                        chunk_dict = dict(chunk)
                        if 'text' in chunk_dict:
                            context_text += f"Resultado {i}:\n"
                            context_text += f"{chunk_dict['text']}\n"
                            
                            # Add relationship information if available
                            if 'r' in result and 'm' in result:
                                context_text += f"Relaciones encontradas: {result.get('r', 'N/A')}\n"
                                context_text += f"Entidades relacionadas: {result.get('m', 'N/A')}\n"
                            context_text += "\n"
                else:
                    context_text += f"Resultado {i} (Formato desconocido):\n"
                    context_text += f"{str(result)}\n\n"
    
    # Create the prompt for Gemini
    system_prompt = f"""Eres un experto en análisis de textos literarios y conocimiento de Don Quijote. 
    Has recibido resultados de un sistema de recuperación de información basado en {"búsqueda vectorial" if method_type == "vector" else "grafos de conocimiento (GraphRAG)"}.
    
    Tu tarea es:
    1. Analizar los resultados de recuperación proporcionados
    2. Responder la pregunta del usuario de manera completa y precisa
    3. {"Explicar las relaciones entre entidades que encuentres en los resultados" if method_type == "cypher" else "Proporcionar información relevante basada en la similitud semántica"}
    4. Proporcionar información adicional relevante basada en el contexto recuperado
    
    Responde en español de manera clara y estructurada."""
    
    user_prompt = f"""
    PREGUNTA DEL USUARIO: {user_question}
    
    {context_text}
    
    Por favor, analiza los resultados de recuperación y responde la pregunta del usuario de manera completa.
    """
    
    # Create messages for Gemini
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Get response from Gemini
    try:
        response = gemini_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error al procesar con Gemini: {e}"

# %%
# Create comprehensive function that combines everything
def complete_graphrag_gemini_workflow(query: str, top_k: int = 3):
    """
    Complete workflow: GraphRAG retrieval + Gemini processing
    """
    print(f"\n{'='*80}")
    print(f"🎯 WORKFLOW COMPLETO PARA: '{query}'")
    print(f"{'='*80}")
    
    # Step 1: Enhanced retrieval
    print("\n1️⃣ RECUPERACIÓN GRAPHRAG:")
    retrieval_results = enhanced_retrieval(query, top_k)
    
    # Step 2: Gemini processing - VECTOR METHOD
    print("\n2️⃣ PROCESAMIENTO CON GEMINI LLM (MÉTODO VECTORIAL):")
    print("-" * 40)
    vector_gemini_response = process_with_gemini(
        retrieval_results["vector_results"], 
        query, 
        method_type="vector"
    )
    
    print("\n📝 RESPUESTA GEMINI (VECTOR):")
    print("=" * 40)
    print(vector_gemini_response)
    print("\n" + "=" * 40)
    
    # Step 3: Gemini processing - CYPHER METHOD
    print("\n3️⃣ PROCESAMIENTO CON GEMINI LLM (MÉTODO CYPHER):")
    print("-" * 40)
    cypher_gemini_response = process_with_gemini(
        retrieval_results["cypher_results"], 
        query, 
        method_type="cypher"
    )
    
    print("\n📝 RESPUESTA GEMINI (CYPHER):")
    print("=" * 40)
    print(cypher_gemini_response)
    print("\n" + "=" * 40)
    
    # Step 4: Display comparison
    print("\n🔍 COMPARACIÓN DE RESPUESTAS:")
    print("=" * 80)
    print("VECTOR vs CYPHER - Misma consulta, diferentes métodos de recuperación")
    print("=" * 80)
    print(f"\n📊 RESPUESTA VECTORIAL:\n{vector_gemini_response}")
    print(f"\n🔗 RESPUESTA CYPHER:\n{cypher_gemini_response}")
    print("\n" + "=" * 80)
    
    # Step 5: Return structured result
    return {
        "query": query,
        "retrieval_results": retrieval_results,
        "vector_gemini_response": vector_gemini_response,
        "cypher_gemini_response": cypher_gemini_response
    }

# %% [markdown]
# ## Cell: Vector and Cypher Retrieval

# %%
# Create function to get vector and cypher results separately
def get_retrieval_results(query: str, top_k: int = 3):
    """
    Get both vector and cypher retrieval results for comparison
    """
    print(f"\n{'='*80}")
    print(f"🔍 OBTENIENDO RESULTADOS DE RECUPERACIÓN PARA: '{query}'")
    print(f"{'='*80}")
    
    # Get vector results only
    print("\n1️⃣ OBTENIENDO RESULTADOS VECTORIALES:")
    print("-" * 40)
    try:
        vector_resp = vector_retriever.get_search_results(query_text=query, top_k=top_k)
        vector_results = []
        for i, record in enumerate(vector_resp.records, 1):
            print(f"  {i}. Score: {record.get('score', 'N/A')}")
            if 'text' in record:
                print(f"     Text: {record['text'][:200]}...")
            vector_results.append(record)
    except Exception as e:
        print(f"  Error: {e}")
        vector_results = []
    
    # Get cypher results only
    print("\n2️⃣ OBTENIENDO RESULTADOS CON RELACIONES (CYPHER):")
    print("-" * 40)
    try:
        vc_resp = vector_cypher_retriever.get_search_results(query_text=query, top_k=top_k)
        cypher_results = []
        if hasattr(vc_resp, 'records'):
            for i, record in enumerate(vc_resp.records, 1):
                print(f"  Record {i}:")
                # Check if this is the new hopsCypherQuery format (returns 'chunks' and 'rels' fields)
                if 'chunks' in record and 'rels' in record:
                    print(f"    Chunks: {len(record['chunks'])} items")
                    print(f"    Relationships: {len(record['rels'])} items")
                    if record['chunks']:
                        print(f"    First chunk text: {record['chunks'][0].get('text', 'No text')[:100]}...")
                    if record['rels']:
                        print(f"    First relationship: {type(record['rels'][0])}")
                    cypher_results.append(record)
                # Check for old 'info' format
                elif 'info' in record:
                    print(f"    Formatted info: {record['info'][:200]}...")
                    cypher_results.append(record)
                # Fallback to old format for backward compatibility
                elif 'chunk' in record:
                    chunk = record.get('chunk')
                    if chunk and hasattr(chunk, '__dict__'):
                        chunk_dict = dict(chunk)
                        if 'text' in chunk_dict:
                            print(f"    Chunk text: {chunk_dict['text'][:150]}...")
                    cypher_results.append(record)
                else:
                    print(f"    Unknown format: {list(record.keys())}")
                    cypher_results.append(record)
        else:
            print(f"  Response: {str(vc_resp)}")
    except Exception as e:
        print(f"  Error: {e}")
        cypher_results = []
    
    return {
        "query": query,
        "vector_results": vector_results,
        "cypher_results": cypher_results
    }

# %% [markdown]
# ## Cell: Process with Gemini LLM
# %% 
# # ## Cell: Process with Gemini LLM
get_retrieval_results("¿Quién es Don Quijote y cuáles son sus aventuras?", top_k=2)

##
# %%
# Create function to process retrieval results with Gemini
def process_retrieval_with_gemini(retrieval_data: dict):
    """
    Process both vector and cypher results with Gemini LLM
    """
    query = retrieval_data["query"]
    vector_results = retrieval_data["vector_results"]
    cypher_results = retrieval_data["cypher_results"]
    
    print(f"\n{'='*80}")
    print(f"🤖 PROCESAMIENTO CON GEMINI LLM PARA: '{query}'")
    print(f"{'='*80}")
    
    # Process with Gemini - VECTOR METHOD
    print("\n3️⃣ PROCESAMIENTO CON GEMINI (MÉTODO VECTORIAL):")
    print("-" * 40)
    vector_gemini_response = process_with_gemini(vector_results, query, method_type="vector")
    
    print("\n📝 RESPUESTA GEMINI (VECTOR):")
    print("=" * 50)
    print(vector_gemini_response)
    print("\n" + "=" * 50)
    
    # Process with Gemini - CYPHER METHOD
    print("\n4️⃣ PROCESAMIENTO CON GEMINI (MÉTODO CYPHER):")
    print("-" * 40)
    cypher_gemini_response = process_with_gemini(cypher_results, query, method_type="cypher")
    
    print("\n📝 RESPUESTA GEMINI (CYPHER):")
    print("=" * 50)
    print(cypher_gemini_response)
    print("\n" + "=" * 50)
    
    # Display comparison
    print("\n🔍 COMPARACIÓN DE RESPUESTAS:")
    print("=" * 80)
    print("VECTOR vs CYPHER - Misma consulta, diferentes métodos de recuperación")
    print("=" * 80)
    print(f"\n📊 RESPUESTA VECTORIAL:\n{vector_gemini_response}")
    print(f"\n🔗 RESPUESTA CYPHER:\n{cypher_gemini_response}")
    print("\n" + "=" * 80)
    
    return {
        "query": query,
        "vector_results": vector_results,
        "cypher_results": cypher_results,
        "vector_gemini_response": vector_gemini_response,
        "cypher_gemini_response": cypher_gemini_response
    }

# %% [markdown]
# ## Cell: Usage Example - Split Workflow

# %%
# Example usage of the split workflow
def run_split_workflow(query: str, top_k: int = 3):
    """
    Example of how to use the split workflow:
    1. First get retrieval results
    2. Then process with Gemini
    """
    print(f"\n{'='*80}")
    print(f"🚀 EJEMPLO DE WORKFLOW DIVIDIDO PARA: '{query}'")
    print(f"{'='*80}")
    
    # Step 1: Get retrieval results
    print("\n📥 PASO 1: Obtener resultados de recuperación")
    retrieval_data = get_retrieval_results(query, top_k)
    
    # Step 2: Process with Gemini
    print("\n🤖 PASO 2: Procesar con Gemini LLM")
    final_results = process_retrieval_with_gemini(retrieval_data)
    
    return final_results

# %%
# Test the complete workflow with the test queries
print("\n" + "="*80)
print("🚀 PROBANDO WORKFLOW COMPLETO: GraphRAG + Gemini LLM")
print("="*80)

# Test with our existing queries
for i, query in enumerate(test_queries, 1):
    print(f"\n{'#'*80}")
    print(f"CONSULTA {i} DE {len(test_queries)}")
    print(f"{'#'*80}")
    
    result = complete_graphrag_gemini_workflow(query, top_k=2)
    
    print(f"\n✅ Consulta {i} completada exitosamente!")
    print(f"{'#'*80}\n")

print("\n" + "="*80)
print("🎉 ¡TODAS LAS CONSULTAS PROCESADAS EXITOSAMENTE!")
print("="*80)
print("\nEste ejemplo demuestra la integración completa de:")
print("- Recuperación vectorial simple")
print("- Recuperación vectorial con relaciones de grafo (Cypher)")
print("- Procesamiento avanzado con Gemini LLM")
print("- Respuestas enriquecidas con contexto de grafos de conocimiento")
print("\n" + "="*80)

# %%
# Test the comparison method separately
print("\n" + "="*80)
print("🔬 PROBANDO MÉTODO DE COMPARACIÓN: Vector vs Cypher")
print("="*80)

# Test with a specific query for detailed comparison
comparison_query = "¿Quién es Don Quijote y cuáles son sus aventuras?"
print(f"\n🎯 Consulta de comparación: '{comparison_query}'")

comparison_result = compare_retrieval_methods(comparison_query, top_k=2)

print(f"\n✅ Comparación completada exitosamente!")
print(f"📊 Resultados vectoriales: {len(comparison_result['vector_results'])}")
print(f"🔗 Resultados cypher: {len(comparison_result['cypher_results'])}")
print(f"📝 Respuesta vectorial generada: {'Sí' if comparison_result['vector_gemini_response'] else 'No'}")
print(f"📝 Respuesta cypher generada: {'Sí' if comparison_result['cypher_gemini_response'] else 'No'}")

print("\n" + "="*80)
print("🎉 ¡COMPARACIÓN DE MÉTODOS COMPLETADA!")
print("="*80)
print("\nEste ejemplo demuestra:")
print("- Recuperación independiente con cada método")
print("- Comparación directa de resultados")
print("- Análisis de ventajas/desventajas de cada enfoque")
print("- Identificación de relaciones adicionales en Cypher")
print("\n" + "="*80)

# %%


# %%
vector_retriever = VectorRetriever(neo4j_driver, "text_embeddings", embedder)
retrieval_query_2 = """
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)-[r]->(child)-[rr]->(child2)
RETURN 
  similarity_score, 
  apoc.map.removeKeys(properties(chunk), ['embedding', 'query_vector']) AS chunk,
  apoc.map.removeKeys(properties(child), ['embedding', 'query_vector']) AS child
"""
cypher_retriever = VectorCypherRetriever(neo4j_driver, "text_embeddings", retrieval_query_2, embedder)

query = "¿Quién es Don Quijote?"
vector_results = vector_retriever.get_search_results(query_text=query, top_k=3)
cypher_results = cypher_retriever.get_search_results(query_text=query, top_k=3)

def clean_results(results):
    # Remove query_vector from results metadata
    if hasattr(results, 'metadata') and 'query_vector' in results.metadata:
        del results.metadata['query_vector']
    
    # Also clean individual records if needed
    if hasattr(results, 'records'):
        for record in results.records:
            # Check metadata
            if hasattr(record, 'metadata') and 'query_vector' in record.metadata:
                del record.metadata['query_vector']
            # Check if record itself has query_vector
            if hasattr(record, 'query_vector'):
                delattr(record, 'query_vector')
            # Check if record is a dict-like object
            if isinstance(record, dict) and 'query_vector' in record:
                del record['query_vector']
    return results

vector_results = clean_results(vector_results)
cypher_results = clean_results(cypher_results)

print("VECTOR RESULTS:")
print(vector_results)
print("\nCYPHER RESULTS:")
print(cypher_results)

# %%
def process_results(query, results):
    system_prompt = f"""Eres un experto en análisis de textos literarios y conocimiento de Don Quijote. 
Has recibido resultados de un sistema de recuperación de información.

Tu tarea es:
1. Analizar los resultados de recuperación proporcionados
2. Responder la pregunta del usuario de manera completa y precisa
3. Proporcionar información relevante basada en el contexto recuperado

Responde en español de manera clara y estructurada."""

    user_prompt = f"""
PREGUNTA DEL USUARIO: {query}

Retrieval Results: {str(results)}

Por favor, analiza los resultados de recuperación y responde la pregunta del usuario de manera completa.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = gemini_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error al procesar con Gemini: {e}"

response = process_results(query, vector_results)
print(response)

response = process_results(query, cypher_results)
print(response)

# %%
