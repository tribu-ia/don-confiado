import os
import uuid
import asyncio
from typing import Optional, Dict, Any, List
import tempfile

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Import neo4j-graphrag components
try:
    from neo4j_graphrag.llm import OpenAILLM as LLM
    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings as Embeddings
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from neo4j_graphrag.experimental.components.text_splitters.langchain import LangChainTextSplitterAdapter
    NEO4J_GRAPHRAG_AVAILABLE = True
except ImportError:
    NEO4J_GRAPHRAG_AVAILABLE = False

from ai.enhanced_graphrag_config import (
    get_embeddings,
    get_chat_model,
    get_market_research_entities_config,
    get_market_research_relations_config,
    get_market_research_extraction_prompt_config,
    get_kg_builder_config
)

load_dotenv()

# Job tracking
ENHANCED_JOBS: Dict[str, Dict[str, Any]] = {}


def _read_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("pypdf is required for PDF ingestion")
    
    reader = PdfReader(file_path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip()


def _get_driver():
    """Get Neo4j driver connection"""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def _setup_llm_and_embeddings():
    """Setup LLM and embeddings for neo4j-graphrag"""
    if not NEO4J_GRAPHRAG_AVAILABLE:
        raise RuntimeError("neo4j-graphrag not available. Install with: pip install neo4j-graphrag")
    
    # Use OpenAI for neo4j-graphrag compatibility
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not configured for neo4j-graphrag")
    
    llm = LLM(
        model_name="gpt-4o-mini",
        model_params={
            "response_format": {"type": "json_object"},
            "temperature": 0
        }
    )
    
    embedder = Embeddings()
    
    return llm, embedder


async def ingest_with_ontology_async(
    text: str, 
    title: Optional[str] = None,
    ontology_type: str = "football"
) -> str:
    """
    Ingest text using SimpleKGPipeline with ontology-based knowledge extraction
    """
    if not NEO4J_GRAPHRAG_AVAILABLE:
        raise RuntimeError("neo4j-graphrag not available")
    
    job_id = str(uuid.uuid4())
    ENHANCED_JOBS[job_id] = {"status": "running", "ontology": ontology_type}
    
    try:
        # Setup components
        llm, embedder = _setup_llm_and_embeddings()
        driver = _get_driver()
        
        # Get ontology configuration
        if ontology_type == "market_research":
            entities = get_market_research_entities_config()
            relations = get_market_research_relations_config()
            prompt_template = get_market_research_extraction_prompt_config()
        else:
            # Default to market research configuration
            entities = get_market_research_entities_config()
            relations = get_market_research_relations_config()
            prompt_template = get_market_research_extraction_prompt_config()
        
        # Configure text splitter
        config = get_kg_builder_config()
        text_splitter = FixedSizeSplitter(
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["chunk_overlap"]
        )
        
        # Create SimpleKGPipeline
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            text_splitter=text_splitter,
            embedder=embedder,
            entities=entities,
            relations=relations,
            prompt_template=prompt_template,
            from_pdf=False,
            perform_entity_resolution=config["perform_entity_resolution"]
        )
        
        # Run the pipeline
        result = await kg_builder.run_async(text=text)
        
        # Update job status
        ENHANCED_JOBS[job_id] = {
            "status": "completed",
            "ontology": ontology_type,
            "entities": entities,
            "relations": relations,
            "result": result
        }
        
    except Exception as e:
        ENHANCED_JOBS[job_id] = {
            "status": "failed", 
            "error": str(e),
            "ontology": ontology_type
        }
    finally:
        if 'driver' in locals():
            driver.close()
    
    return job_id


def ingest_with_ontology(
    text: str, 
    title: Optional[str] = None,
    ontology_type: str = "market_research"
) -> str:
    """
    Synchronous ontology-based ingestion using SimpleKGPipeline
    """
    if not NEO4J_GRAPHRAG_AVAILABLE:
        raise RuntimeError("neo4j-graphrag not available")
    
    job_id = str(uuid.uuid4())
    ENHANCED_JOBS[job_id] = {"status": "running", "ontology": ontology_type}
    
    try:
        # Setup components
        llm, embedder = _setup_llm_and_embeddings()
        driver = _get_driver()
        
        # Get market research ontology configuration
        entities = get_market_research_entities_config()
        relations = get_market_research_relations_config()
        prompt_template = get_market_research_extraction_prompt_config()
        
        # Create SimpleKGPipeline with market research ontology
        kg_builder = SimpleKGPipeline(
            llm=llm,
            driver=driver,
            text_splitter=FixedSizeSplitter(chunk_size=500, chunk_overlap=100),
            embedder=embedder,
            entities=entities,
            relations=relations,
            prompt_template=prompt_template,
            from_pdf=False,
            perform_entity_resolution=True
        )
        
        # Run the pipeline synchronously using asyncio in a new thread
        import asyncio
        import threading
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(kg_builder.run_async(text=text))
            finally:
                loop.close()
        
        result = run_in_thread()
        
        # Update job status
        ENHANCED_JOBS[job_id] = {
            "status": "completed",
            "ontology": ontology_type,
            "entities": entities,
            "relations": relations,
            "result": result
        }
        
    except Exception as e:
        ENHANCED_JOBS[job_id] = {
            "status": "failed", 
            "error": str(e),
            "ontology": ontology_type
        }
    finally:
        if 'driver' in locals():
            driver.close()
    
    return job_id


def ingest_pdf_with_ontology(
    file_path: str, 
    title: Optional[str] = None,
    ontology_type: str = "market_research"
) -> str:
    """
    Ingest PDF with market research ontology-based knowledge extraction
    """
    text = _read_text_from_pdf(file_path)
    return ingest_with_ontology(text, title or os.path.basename(file_path), ontology_type)


def get_enhanced_job(job_id: str) -> Dict[str, Any]:
    """Get enhanced job status"""
    return ENHANCED_JOBS.get(job_id, {"status": "unknown"})


def get_ontology_stats(ontology_type: str = "football") -> Dict[str, Any]:
    """
    Get statistics about the knowledge graph for a specific ontology
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            # Get entity counts
            entity_query = """
            MATCH (n)
            WHERE n:__Entity__
            RETURN labels(n) as entity_type, count(n) as count
            ORDER BY count DESC
            """
            
            # Get relationship counts  
            rel_query = """
            MATCH ()-[r]->()
            WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK']
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
            """
            
            entity_results = session.run(entity_query)
            rel_results = session.run(rel_query)
            
            entities = [{"type": record["entity_type"], "count": record["count"]} 
                      for record in entity_results]
            relationships = [{"type": record["rel_type"], "count": record["count"]} 
                           for record in rel_results]
            
            return {
                "ontology_type": ontology_type,
                "entities": entities,
                "relationships": relationships,
                "total_entities": sum(e["count"] for e in entities),
                "total_relationships": sum(r["count"] for r in relationships)
            }
            
    finally:
        driver.close()


def clear_ontology_data(ontology_type: str = "football") -> bool:
    """
    Clear all knowledge graph data for a specific ontology
    """
    driver = _get_driver()
    try:
        with driver.session() as session:
            # Clear all entities and relationships
            clear_query = """
            MATCH (n)
            WHERE n:__Entity__ OR n:Chunk OR n:Document
            DETACH DELETE n
            """
            session.run(clear_query)
            return True
    except Exception as e:
        print(f"Error clearing ontology data: {e}")
        return False
    finally:
        driver.close()
