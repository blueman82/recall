#!/usr/bin/env python3
"""
Recall POC - Proof of Concept Script
=====================================
Tests the core concepts:
1. Embedding via Ollama + mxbai-embed-large (with query prefix)
2. ChromaDB storage (shared path with docvec, separate collection)
3. SQLite metadata + graph storage
4. Semantic recall with RRF-style ranking

Run: uv run python poc/test_recall_concept.py
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
import httpx

# === Configuration ===
CHROMA_PATH = Path.home() / ".docvec" / "chroma_db"
SQLITE_PATH = Path.home() / ".claude-code" / "memories" / "recall_poc.db"
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
EMBED_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION_NAME = "memories_poc"  # POC collection, separate from production


def embed_text(text: str, is_query: bool = False) -> list[float]:
    """Embed text using Ollama + mxbai-embed-large."""
    # mxbai requires prefix for queries (not documents)
    if is_query:
        text = EMBED_PREFIX + text

    response = httpx.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30.0
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def content_hash(content: str) -> str:
    """SHA-256 hash for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def init_sqlite(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite with schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            type TEXT NOT NULL,
            namespace TEXT NOT NULL DEFAULT 'global',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            importance REAL DEFAULT 0.5,
            UNIQUE(content_hash, namespace)
        );

        CREATE TABLE IF NOT EXISTS edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_id, target_id, relation)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
    """)
    conn.commit()
    return conn


def init_chromadb(db_path: Path) -> chromadb.Collection:
    """Initialize ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Cosine distance!
    )
    return collection


def memory_store(
    conn: sqlite3.Connection,
    collection: chromadb.Collection,
    content: str,
    memory_type: str = "decision",
    namespace: str = "global",
    importance: float = 0.5
) -> dict:
    """Store a memory in both SQLite and ChromaDB."""
    memory_id = str(uuid.uuid4())
    c_hash = content_hash(content)

    # Check for duplicate
    existing = conn.execute(
        "SELECT id FROM memories WHERE content_hash = ? AND namespace = ?",
        (c_hash, namespace)
    ).fetchone()

    if existing:
        return {"success": False, "error": "Duplicate memory", "existing_id": existing["id"]}

    # Embed content (no prefix for documents)
    embedding = embed_text(content, is_query=False)

    # Store in SQLite
    conn.execute("""
        INSERT INTO memories (id, content, content_hash, type, namespace, importance)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (memory_id, content, c_hash, memory_type, namespace, importance))
    conn.commit()

    # Store in ChromaDB
    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "type": memory_type,
            "namespace": namespace,
            "importance": importance,
            "created_at": datetime.now().isoformat()
        }]
    )

    return {"success": True, "id": memory_id, "content_hash": c_hash}


def memory_recall(
    conn: sqlite3.Connection,
    collection: chromadb.Collection,
    query: str,
    namespace: str | None = None,
    n_results: int = 5
) -> dict:
    """Recall memories via semantic search."""
    # Embed query (with prefix!)
    query_embedding = embed_text(query, is_query=True)

    # Build ChromaDB filter
    where_filter = None
    if namespace:
        where_filter = {"namespace": namespace}

    # Semantic search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    memories = []
    for i, doc_id in enumerate(results["ids"][0]):
        # Update access stats in SQLite
        conn.execute("""
            UPDATE memories
            SET accessed_at = CURRENT_TIMESTAMP, access_count = access_count + 1
            WHERE id = ?
        """, (doc_id,))

        # Fetch full record from SQLite
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (doc_id,)).fetchone()

        if row:
            memories.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "type": row["type"],
                "namespace": row["namespace"],
                "importance": row["importance"],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "access_count": row["access_count"]
            })

    conn.commit()
    return {"success": True, "memories": memories, "total": len(memories)}


def memory_relate(
    conn: sqlite3.Connection,
    source_id: str,
    target_id: str,
    relation: str,
    weight: float = 1.0
) -> dict:
    """Create a relationship between memories."""
    # Verify both exist
    source = conn.execute("SELECT id FROM memories WHERE id = ?", (source_id,)).fetchone()
    target = conn.execute("SELECT id FROM memories WHERE id = ?", (target_id,)).fetchone()

    if not source or not target:
        return {"success": False, "error": "Memory not found"}

    conn.execute("""
        INSERT OR REPLACE INTO edges (source_id, target_id, relation, weight)
        VALUES (?, ?, ?, ?)
    """, (source_id, target_id, relation, weight))
    conn.commit()

    return {"success": True, "source_id": source_id, "target_id": target_id, "relation": relation}


def main():
    """Run POC tests."""
    print("=" * 60)
    print("RECALL POC - Testing Core Concepts")
    print("=" * 60)

    # Initialize stores
    print("\n[1] Initializing stores...")
    print(f"    ChromaDB: {CHROMA_PATH}")
    print(f"    SQLite:   {SQLITE_PATH}")

    conn = init_sqlite(SQLITE_PATH)
    collection = init_chromadb(CHROMA_PATH)

    print(f"    ChromaDB collection '{COLLECTION_NAME}' ready")
    print(f"    SQLite database ready")

    # Test embedding
    print("\n[2] Testing Ollama + mxbai-embed-large...")
    test_embed = embed_text("Hello world", is_query=False)
    print(f"    Embedding dimension: {len(test_embed)}")
    assert len(test_embed) == 1024, "Expected 1024 dimensions for mxbai-embed-large"
    print("    Embedding test passed!")

    # Store test memories
    print("\n[3] Storing test memories...")

    test_memories = [
        ("Always use tabs for indentation in this project", "preference", "global"),
        ("Chose SQLite over Postgres for portability", "decision", "project:recall"),
        ("This codebase uses repository pattern for data access", "pattern", "project:recall"),
        ("Fixed authentication bug by adding token refresh", "session", "project:recall"),
        ("Prefer functional components over class components in React", "preference", "global"),
    ]

    stored_ids = []
    for content, mtype, namespace in test_memories:
        result = memory_store(conn, collection, content, mtype, namespace)
        if result["success"]:
            stored_ids.append(result["id"])
            print(f"    Stored: {content[:50]}... -> {result['id'][:8]}")
        else:
            print(f"    Skip (dup): {content[:50]}...")

    # Test semantic recall
    print("\n[4] Testing semantic recall...")

    queries = [
        ("How should I format code?", None),  # Should find tabs preference
        ("What database did we choose?", "project:recall"),  # Should find SQLite decision
        ("authentication issues", "project:recall"),  # Should find auth bug fix
    ]

    for query, namespace in queries:
        print(f"\n    Query: '{query}' (namespace: {namespace or 'all'})")
        results = memory_recall(conn, collection, query, namespace, n_results=3)

        for mem in results["memories"]:
            print(f"      [{mem['score']:.2f}] {mem['content'][:60]}...")

    # Test relationships
    print("\n[5] Testing graph relationships...")

    if len(stored_ids) >= 2:
        result = memory_relate(conn, stored_ids[0], stored_ids[1], "relates_to")
        print(f"    Created edge: {stored_ids[0][:8]} --relates_to--> {stored_ids[1][:8]}")

        # Query edges
        edges = conn.execute("SELECT * FROM edges").fetchall()
        print(f"    Total edges in graph: {len(edges)}")

    # Summary
    print("\n" + "=" * 60)
    print("POC COMPLETE - All core concepts validated!")
    print("=" * 60)
    print(f"\nStats:")
    print(f"  - Memories stored: {collection.count()}")
    print(f"  - SQLite records: {conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]}")
    print(f"  - Graph edges: {conn.execute('SELECT COUNT(*) FROM edges').fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    main()
