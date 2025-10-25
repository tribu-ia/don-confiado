import os
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver


load_dotenv()


def get_neo4j_driver() -> Driver:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not password:
        raise RuntimeError("Missing NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD environment variables")
    return GraphDatabase.driver(uri, auth=(user, password))


def verify_connection() -> bool:
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS ok")
            record = result.single()
            return bool(record and record.get("ok") == 1)
    finally:
        driver.close()


def ensure_vector_index(index_name: str, dimensions: int, similarity: str = "cosine") -> None:
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            idx_exists = session.run(
                """
                SHOW INDEXES YIELD name
                WHERE name = $index_name
                RETURN name
                """,
                {"index_name": index_name},
            ).single()

            if not idx_exists:
                session.run(
                    f"""
                    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                    FOR (c:Chunk)
                    ON c.embedding
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dimensions},
                            `vector.similarity_function`: '{similarity}'
                        }}
                    }}
                    """
                )
    finally:
        driver.close()


