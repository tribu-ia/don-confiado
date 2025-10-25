import os
import uuid
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from ai.graphrag_config import (
    get_entities,
    get_relations,
    get_extraction_prompt,
    get_embeddings,
)

# Simple ingestion placeholders using neo4j-graphrag style components.
# For this first version, we will only create Document/Chunk nodes and store embeddings.

load_dotenv()


JOBS: Dict[str, Dict[str, Any]] = {}


def _read_text_from_pdf(file_path: str) -> str:
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


def _split_text(text: str, chunk_size: int = 500, overlap: int = 100):
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _get_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def ingest_text(text: str, title: Optional[str] = None) -> str:
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "running"}

    embeddings = get_embeddings()
    chunks = _split_text(text, chunk_size=500, overlap=100)
    vectors = embeddings.embed_documents(chunks) if chunks else []

    driver = _get_driver()
    try:
        with driver.session() as session:
            doc_id = str(uuid.uuid4())
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title = $title, d.created_at = datetime()
                """,
                {"doc_id": doc_id, "title": title or "Uploaded Document"},
            )

            for idx, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
                session.run(
                    """
                    MERGE (c:Chunk {id: $id})
                    SET c.text = $text,
                        c.index = $idx,
                        c.embedding = $embedding,
                        c.created_at = datetime()
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c)-[:FROM_DOCUMENT]->(d)
                    """,
                    {
                        "id": f"{doc_id}:{idx}",
                        "text": chunk_text,
                        "idx": idx,
                        "embedding": vec,
                        "doc_id": doc_id,
                    },
                )

            # Link sequential chunks
            if len(chunks) > 1:
                for idx in range(len(chunks) - 1):
                    session.run(
                        """
                        MATCH (c1:Chunk {id: $id1}), (c2:Chunk {id: $id2})
                        MERGE (c1)-[:NEXT_CHUNK]->(c2)
                        """,
                        {"id1": f"{doc_id}:{idx}", "id2": f"{doc_id}:{idx+1}"},
                    )

        JOBS[job_id] = {"status": "completed", "chunks": len(chunks)}
    except Exception as e:
        JOBS[job_id] = {"status": "failed", "error": str(e)}
    finally:
        driver.close()

    return job_id


def ingest_pdf(file_path: str, title: Optional[str] = None) -> str:
    text = _read_text_from_pdf(file_path)
    return ingest_text(text, title=title or os.path.basename(file_path))


def get_job(job_id: str) -> Dict[str, Any]:
    return JOBS.get(job_id, {"status": "unknown"})


