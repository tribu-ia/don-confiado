"""
Data processing utilities for Neo4j natural language query results.
Extracts and formats data for use in report generation.
"""

import re
import json
from typing import List, Dict, Any, Optional


def parse_record_content(record_str: str) -> Dict[str, Any]:
    """
    Parse a Neo4j Record string into structured data.
    
    Record format example:
    <Record info='=== text ===\n...text...\n=== kg_rels ===\n...relationships...'>
    
    Returns:
        Dict with 'text', 'relationships', and 'raw' keys
    """
    if not record_str or not isinstance(record_str, str):
        return {"text": "", "relationships": [], "raw": record_str}
    
    result = {
        "text": "",
        "relationships": [],
        "raw": record_str
    }
    
    # Extract content between quotes
    match = re.search(r"info='(.*?)'", record_str, re.DOTALL)
    if not match:
        # Try to extract any text content
        result["text"] = record_str[:500]  # Limit length
        return result
    
    content = match.group(1)
    
    # Split by === text === and === kg_rels ===
    text_match = re.search(r'=== text ===\s*(.*?)(?=\s*=== kg_rels ===|$)', content, re.DOTALL)
    if text_match:
        result["text"] = text_match.group(1).strip()
    
    # Extract relationships
    rels_match = re.search(r'=== kg_rels ===\s*(.*?)$', content, re.DOTALL)
    if rels_match:
        rels_text = rels_match.group(1).strip()
        # Parse relationships (format: "Node1 - REL_TYPE() -> Node2")
        relationships = []
        for line in rels_text.split('\n'):
            line = line.strip()
            if line and line != 'null':
                # Extract relationship pattern
                rel_match = re.match(r'(.+?)\s*-\s*(\w+)\(\)\s*->\s*(.+)', line)
                if rel_match:
                    relationships.append({
                        "from": rel_match.group(1).strip(),
                        "type": rel_match.group(2).strip(),
                        "to": rel_match.group(3).strip()
                    })
                else:
                    # Store as raw relationship if pattern doesn't match
                    relationships.append({"raw": line})
        result["relationships"] = relationships
    
    return result


def process_natural_language_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process natural language query results into a structured format
    that's easy for the LLM to use in report generation.
    
    Args:
        results: List of results from neo4j_natural_language_query
        
    Returns:
        Dict with structured data:
        {
            "summary": "Brief summary of findings",
            "text_content": ["extracted text chunks"],
            "relationships": [list of relationships],
            "entities": [list of entities mentioned],
            "raw_results": original results
        }
    """
    if not results:
        return {
            "summary": "No data found in Neo4j graph.",
            "text_content": [],
            "relationships": [],
            "entities": [],
            "raw_results": []
        }
    
    all_text = []
    all_relationships = []
    entities_set = set()
    
    for result in results:
        content = result.get("content", "")
        
        # Parse Record content if it's a string
        if isinstance(content, str) and "Record info" in content:
            parsed = parse_record_content(content)
            if parsed["text"]:
                all_text.append(parsed["text"])
            if parsed["relationships"]:
                all_relationships.extend(parsed["relationships"])
        elif isinstance(content, dict):
            # Already structured
            if "text" in content:
                all_text.append(content["text"])
            if "relationships" in content:
                all_relationships.extend(content["relationships"])
        elif isinstance(content, str):
            # Plain text
            all_text.append(content)
    
    # Extract entities from relationships
    for rel in all_relationships:
        if isinstance(rel, dict):
            if "from" in rel:
                entities_set.add(rel["from"])
            if "to" in rel:
                entities_set.add(rel["to"])
    
    # Extract entities from text (simple extraction of capitalized words/phrases)
    for text in all_text:
        # Look for capitalized words/phrases (potential entities)
        entities = re.findall(r'\b[A-Z][a-zA-Z\s]+', text)
        entities_set.update([e.strip() for e in entities if len(e.strip()) > 2])
    
    # Create summary
    summary_parts = []
    if all_text:
        summary_parts.append(f"Found {len(all_text)} relevant text chunks from the graph.")
    if all_relationships:
        summary_parts.append(f"Identified {len(all_relationships)} relationships.")
    if entities_set:
        summary_parts.append(f"Discovered {len(entities_set)} entities: {', '.join(list(entities_set)[:10])}.")
    
    summary = " ".join(summary_parts) if summary_parts else "Retrieved data from Neo4j graph."
    
    return {
        "summary": summary,
        "text_content": all_text[:10],  # Limit to top 10
        "relationships": all_relationships[:20],  # Limit to top 20
        "entities": list(entities_set)[:15],  # Limit to top 15
        "raw_results": results
    }


def format_neo4j_data_for_llm(processed_data: Dict[str, Any]) -> str:
    """
    Format processed Neo4j data into a readable string for the LLM.
    
    Args:
        processed_data: Output from process_natural_language_results
        
    Returns:
        Formatted string that the LLM can easily understand
    """
    parts = []
    
    # Summary
    parts.append(f"RESUMEN: {processed_data['summary']}")
    
    # Text content
    if processed_data["text_content"]:
        parts.append("\nCONTENIDO DE TEXTO:")
        for i, text in enumerate(processed_data["text_content"][:5], 1):
            # Clean and truncate text
            clean_text = re.sub(r'\s+', ' ', text).strip()[:300]
            parts.append(f"  {i}. {clean_text}...")
    
    # Relationships
    if processed_data["relationships"]:
        parts.append("\nRELACIONES ENCONTRADAS:")
        for i, rel in enumerate(processed_data["relationships"][:10], 1):
            if isinstance(rel, dict):
                if "from" in rel and "to" in rel:
                    parts.append(f"  {i}. {rel['from']} --[{rel.get('type', 'RELATED_TO')}]--> {rel['to']}")
                elif "raw" in rel:
                    parts.append(f"  {i}. {rel['raw']}")
    
    # Entities
    if processed_data["entities"]:
        parts.append(f"\nENTIDADES MENCIONADAS: {', '.join(processed_data['entities'][:10])}")
    
    return "\n".join(parts)

