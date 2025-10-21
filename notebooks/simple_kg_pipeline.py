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

# %%
# Import and setup Vector Cypher Retriever
from neo4j_graphrag.retrievers import VectorCypherRetriever

# Define a simple cypher query for retrieval
simple_cypher_query = """
CALL db.index.vector.queryNodes('text_embeddings', $top_k, $query_vector)
YIELD node AS chunk, score AS similarity_score
MATCH (chunk)-[r]->(m)
RETURN chunk, r, m, similarity_score
"""

# Initialize Vector Cypher Retriever
print("Initializing Vector Cypher Retriever...")

vector_cypher_retriever = VectorCypherRetriever(
    neo4j_driver,
    index_name="text_embeddings",
    embedder=embedder,
    retrieval_query=simple_cypher_query
)
print("✅ VectorCypherRetriever initialized successfully!")

print("\nTesting Vector Cypher Retriever...")
print("=" * 60)

for i, query in enumerate(test_queries, 1):
    print(f"\n🔍 QUERY {i}: {query}")
    print("-" * 50)
    
    # Vector Cypher Retriever
    print("\n🔗 VECTOR CYPHER RETRIEVER RESULTS:")
    try:
        vc_resp = vector_cypher_retriever.get_search_results(query_text=query, top_k=3)
        
        if hasattr(vc_resp, 'records'):
            for i, record in enumerate(vc_resp.records, 1):
                print(f"  Record {i}: {record}")
                n = record.get('n')
                m = record.get('m')
                
                # Extract text using dict conversion
                if n and hasattr(n, '__dict__'):
                    n_dict = dict(n)
                    if 'text' in n_dict:
                        print(f"    Node {n.element_id}: {n_dict['text'][:200]}...")
                
                if m and hasattr(m, '__dict__'):
                    m_dict = dict(m)
                    if 'text' in m_dict:
                        print(f"    Node {m.element_id}: {m_dict['text'][:200]}...")
        else:
            print(f"  Response: {str(vc_resp)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)

print("\n✅ Vector Cypher retrieval testing completed!")
# %%
