"""
Database utilities for DuckDB connection and queries.
"""
import duckdb
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def get_connection(db_path: str = ":memory:"):
    """
    Create a DuckDB connection.
    
    Args:
        db_path: Path to database file, or ":memory:" for in-memory database
    
    Returns:
        DuckDB connection object
    """
    return duckdb.connect(db_path)


def load_csv_to_table(conn, csv_path: str, table_name: str):
    """
    Load a CSV file into a DuckDB table.
    
    Args:
        conn: DuckDB connection
        csv_path: Path to the CSV file
        table_name: Name for the table
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} AS 
        SELECT * FROM read_csv_auto('{csv_path}')
    """)
    print(f"✅ Loaded {csv_path} into table '{table_name}'")


def run_query(conn, query: str):
    """
    Run a SQL query and return results as a pandas DataFrame.
    
    Args:
        conn: DuckDB connection
        query: SQL query string
    
    Returns:
        pandas DataFrame with results
    """
    return conn.execute(query).fetchdf()


def show_tables(conn):
    """Show all tables in the database."""
    return run_query(conn, "SHOW TABLES")


def describe_table(conn, table_name: str):
    """Show schema of a table."""
    return run_query(conn, f"DESCRIBE {table_name}")
