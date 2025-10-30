"""
Example: Using PostgresSqlConnector with pgvector for semantic search

This example demonstrates how to:
1. Create a table with vector columns for embeddings
2. Store embeddings generated from text
3. Perform vector similarity search

Prerequisites:
- PostgreSQL with pgvector extension installed
- Run: CREATE EXTENSION vector;
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import (
    PostgresSqlConnector,
    PostgresDataType
)


@dataclass
class Product:
    """Product with text embedding for semantic search"""
    id: Optional[int] = field(default=None, metadata={"is_pkey": True})
    name: str = field(default="")
    description: str = field(default="")
    metadata: dict = field(default_factory=dict)
    # Vector embedding (1536 dimensions for OpenAI ada-002)
    # Note: In practice, you'd use PostgresDataType.vector(1536) in CREATE TABLE
    embedding: List[float] = field(default_factory=list)
    created_at: Optional[datetime] = field(
        default=None,
        metadata={"auto_time_stamp": True}
    )


def create_products_table_with_vectors(connector: PostgresSqlConnector):
    """
    Create a products table with vector column for embeddings.

    This manually creates the table with proper vector type.
    """
    # Drop existing table
    connector.drop_table("products")

    # Create table with vector column
    sql = f"""
    CREATE TABLE IF NOT EXISTS {connector.schema}.products (
        id SERIAL PRIMARY KEY,
        name {PostgresDataType.VARCHAR},
        description {PostgresDataType.TEXT},
        metadata {PostgresDataType.JSONB},
        embedding {PostgresDataType.vector(1536)},
        created_at {PostgresDataType.TIMESTAMP} DEFAULT CURRENT_TIMESTAMP
    );
    """

    connector.exec_sql(sql)
    print("✓ Created products table with vector column")


def create_vector_index(connector: PostgresSqlConnector):
    """
    Create an IVFFlat index for fast similarity search.

    Index types for pgvector:
    - ivfflat: Approximate nearest neighbor (ANN) - faster, less accurate
    - hnsw: Hierarchical Navigable Small World - more accurate, slower build
    """
    sql = f"""
    CREATE INDEX IF NOT EXISTS products_embedding_idx
    ON {connector.schema}.products
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    """

    connector.exec_sql(sql)
    print("✓ Created vector similarity index")


def insert_product_with_embedding(
    connector: PostgresSqlConnector,
    name: str,
    description: str,
    embedding: List[float],
    metadata: dict = None
):
    """
    Insert a product with its text embedding.

    Args:
        connector: PostgreSQL connector
        name: Product name
        description: Product description
        embedding: Vector embedding (list of floats)
        metadata: Additional metadata (stored as JSONB)
    """
    import json

    # Convert embedding to PostgreSQL array format
    embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
    metadata_json = json.dumps(metadata or {})

    sql = f"""
    INSERT INTO {connector.schema}.products (name, description, metadata, embedding)
    VALUES (%s, %s, %s, %s::vector)
    RETURNING id;
    """

    # Note: This is a simplified version. In production, use parameterized queries
    # For this example, we'll use exec_sql with formatted string (not recommended for production)
    sql_formatted = f"""
    INSERT INTO {connector.schema}.products (name, description, metadata, embedding)
    VALUES ('{name}', '{description}', '{metadata_json}'::jsonb, '{embedding_str}'::vector)
    RETURNING id;
    """

    result = connector.run_sql_and_get_df(sql_formatted)
    product_id = result.iloc[0, 0] if not result.empty else None

    print(f"✓ Inserted product '{name}' with ID {product_id}")
    return product_id


def search_similar_products(
    connector: PostgresSqlConnector,
    query_embedding: List[float],
    limit: int = 5
):
    """
    Find products similar to the query using cosine similarity.

    Args:
        connector: PostgreSQL connector
        query_embedding: Query vector embedding
        limit: Number of results to return

    Returns:
        DataFrame with similar products and their similarity scores
    """
    embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

    sql = f"""
    SELECT
        id,
        name,
        description,
        metadata,
        1 - (embedding <=> '{embedding_str}'::vector) as similarity
    FROM {connector.schema}.products
    ORDER BY embedding <=> '{embedding_str}'::vector
    LIMIT {limit};
    """

    results = connector.run_sql_and_get_df(sql)
    print(f"✓ Found {len(results)} similar products")

    return results


def example_usage():
    """
    Example usage of vector database with PostgreSQL and pgvector.
    """
    # Configure database connection
    db_config = {
        "dbname": "your_database",
        "user": "your_user",
        "password": "your_password",
        "host": "localhost",
        "port": 5432,
        "schema": "public"
    }

    # Create connector
    connector = PostgresSqlConnector(db_config)

    # 1. Create table with vector column
    create_products_table_with_vectors(connector)

    # 2. Create vector index for fast similarity search
    create_vector_index(connector)

    # 3. Insert products with embeddings
    # (In practice, generate these using LangChainEmbeddingsConnector)
    products = [
        {
            "name": "Laptop",
            "description": "High-performance laptop for developers",
            "embedding": [0.1] * 1536,  # Placeholder embedding
            "metadata": {"category": "Electronics", "price": 1299.99}
        },
        {
            "name": "Desk",
            "description": "Ergonomic standing desk",
            "embedding": [0.2] * 1536,  # Placeholder embedding
            "metadata": {"category": "Furniture", "price": 599.99}
        },
    ]

    for product in products:
        insert_product_with_embedding(
            connector,
            product["name"],
            product["description"],
            product["embedding"],
            product["metadata"]
        )

    # 4. Search for similar products
    query_embedding = [0.15] * 1536  # Placeholder query embedding
    results = search_similar_products(connector, query_embedding, limit=5)

    print("\nSimilar products:")
    print(results[['name', 'similarity']].to_string(index=False))


if __name__ == "__main__":
    print("pgvector Example")
    print("=" * 70)
    print("\nNote: This is a demonstration. Update db_config with your credentials.")
    print("\nKey PostgreSQL Vector Operations:")
    print("  • <-> : Cosine distance (lower is more similar)")
    print("  • <#> : Negative inner product")
    print("  • <=> : Euclidean distance")
    print("\n" + "=" * 70)

    # Uncomment to run:
    # example_usage()
