from datetime import datetime
import json
from dataclasses import fields, asdict, dataclass
from typing import Type, Dict, List, Any, Optional
from enum import Enum

from zervedataplatform.abstractions.connectors.SqlConnector import SqlConnector, SqlType

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from zervedataplatform.utils.Utility import Utility


class PGVectorIndexes(Enum):
    """PostgreSQL pgvector index types"""
    IVFFLAT_COSINE = "ivfflat_cosine"
    IVFFLAT_L2 = "ivfflat_l2"
    IVFFLAT_IP = "ivfflat_ip"  # Inner product
    HNSW_COSINE = "hnsw_cosine"
    HNSW_L2 = "hnsw_l2"
    HNSW_IP = "hnsw_ip"

@dataclass(frozen=True)
class PostgresSqlTypeDef(SqlType):
    """
    Represents a PostgreSQL data type with optional parameters.

    Inherits from SqlType base class to provide PostgreSQL-specific type system.

    """
    base_type: str
    length: Optional[int] = None
    precision: Optional[int] = None
    scale: Optional[int] = None
    dimensions: Optional[int] = None

    def to_sql(self) -> str:
        """
        Generate the SQL type string for this type.

        Subclasses may override this method for database-specific formatting.

        Returns:
            str: Valid SQL type declaration
        """
        if self.dimensions is not None:
            return f"{self.base_type}({self.dimensions})"
        elif self.length is not None:
            return f"{self.base_type}({self.length})"
        elif self.precision is not None and self.scale is not None:
            return f"{self.base_type}({self.precision},{self.scale})"
        elif self.precision is not None:
            return f"{self.base_type}({self.precision})"
        return self.base_type

@dataclass(frozen=True)
class PostgresSqlVectorTypeDef(SqlType):
    """
    Represents a PostgreSQL vector index definition for pgvector.

    Supports IVFFlat and HNSW indexes with different distance operators.

    Attributes:
        base_type: Index type identifier
        index_type: PGVectorIndexes enum value
        idx_table: Table name for the index
        idx_column: Column name for the index
        idx_lists: Number of inverted lists (IVFFlat only)
        idx_m: Max number of connections per layer (HNSW only)
        idx_ef_construction: Size of dynamic candidate list for construction (HNSW only)
    """
    base_type: str
    index_type: PGVectorIndexes
    idx_table: Optional[str] = None
    idx_column: Optional[str] = None

    # IVFFlat parameters
    idx_lists: Optional[int] = None

    # HNSW parameters
    idx_m: Optional[int] = None
    idx_ef_construction: Optional[int] = None

    def to_sql(self) -> str:
        """
        Generate the CREATE INDEX SQL statement.

        Returns:
            str: Valid PostgreSQL CREATE INDEX statement for pgvector
        """
        if not self.idx_table or not self.idx_column:
            return self.base_type

        # Determine index method and operator class
        if self.index_type == PGVectorIndexes.IVFFLAT_COSINE:
            index_name = f"{self.idx_table}_{self.idx_column}_ivfflat_cosine_idx"
            params = f"WITH (lists = {self.idx_lists or 100})"
            return (
                f"CREATE INDEX {index_name} "
                f"ON {self.idx_table} USING ivfflat ({self.idx_column} vector_cosine_ops) "
                f"{params};"
            )

        elif self.index_type == PGVectorIndexes.IVFFLAT_L2:
            index_name = f"{self.idx_table}_{self.idx_column}_ivfflat_l2_idx"
            params = f"WITH (lists = {self.idx_lists or 100})"
            return (
                f"CREATE INDEX {index_name} "
                f"ON {self.idx_table} USING ivfflat ({self.idx_column} vector_l2_ops) "
                f"{params};"
            )

        elif self.index_type == PGVectorIndexes.IVFFLAT_IP:
            index_name = f"{self.idx_table}_{self.idx_column}_ivfflat_ip_idx"
            params = f"WITH (lists = {self.idx_lists or 100})"
            return (
                f"CREATE INDEX {index_name} "
                f"ON {self.idx_table} USING ivfflat ({self.idx_column} vector_ip_ops) "
                f"{params};"
            )

        elif self.index_type == PGVectorIndexes.HNSW_COSINE:
            index_name = f"{self.idx_table}_{self.idx_column}_hnsw_cosine_idx"
            m = self.idx_m or 16
            ef = self.idx_ef_construction or 64
            params = f"WITH (m = {m}, ef_construction = {ef})"
            return (
                f"CREATE INDEX {index_name} "
                f"ON {self.idx_table} USING hnsw ({self.idx_column} vector_cosine_ops) "
                f"{params};"
            )

        elif self.index_type == PGVectorIndexes.HNSW_L2:
            index_name = f"{self.idx_table}_{self.idx_column}_hnsw_l2_idx"
            m = self.idx_m or 16
            ef = self.idx_ef_construction or 64
            params = f"WITH (m = {m}, ef_construction = {ef})"
            return (
                f"CREATE INDEX {index_name} "
                f"ON {self.idx_table} USING hnsw ({self.idx_column} vector_l2_ops) "
                f"{params};"
            )

        elif self.index_type == PGVectorIndexes.HNSW_IP:
            index_name = f"{self.idx_table}_{self.idx_column}_hnsw_ip_idx"
            m = self.idx_m or 16
            ef = self.idx_ef_construction or 64
            params = f"WITH (m = {m}, ef_construction = {ef})"
            return (
                f"CREATE INDEX {index_name} "
                f"ON {self.idx_table} USING hnsw ({self.idx_column} vector_ip_ops) "
                f"{params};"
            )

        return self.base_type

# ============================================================================
# Factory functions for parameterized types
# ============================================================================

# For backwards compatibility, keep the old PostgresDataType name as an alias
class PostgresDataType:
    # Integer types
    INTEGER = PostgresSqlTypeDef("INTEGER")
    BIGINT = PostgresSqlTypeDef("BIGINT")
    SMALLINT = PostgresSqlTypeDef("SMALLINT")
    SERIAL = PostgresSqlTypeDef("SERIAL")
    BIGSERIAL = PostgresSqlTypeDef("BIGSERIAL")

    # String types
    VARCHAR = PostgresSqlTypeDef("VARCHAR")  # Generic VARCHAR, use VARCHAR(n) for specific length
    TEXT = PostgresSqlTypeDef("TEXT")
    CHAR = PostgresSqlTypeDef("CHAR")

    # Floating point types
    FLOAT = PostgresSqlTypeDef("FLOAT")
    REAL = PostgresSqlTypeDef("REAL")
    DOUBLE_PRECISION = PostgresSqlTypeDef("DOUBLE PRECISION")
    NUMERIC = PostgresSqlTypeDef("NUMERIC")  # Generic NUMERIC, use NUMERIC(p,s) for specific
    DECIMAL = PostgresSqlTypeDef("DECIMAL")

    # Boolean
    BOOLEAN = PostgresSqlTypeDef("BOOLEAN")

    # Date/Time types
    DATE = PostgresSqlTypeDef("DATE")
    TIME = PostgresSqlTypeDef("TIME")
    TIMESTAMP = PostgresSqlTypeDef("TIMESTAMP")
    TIMESTAMPTZ = PostgresSqlTypeDef("TIMESTAMPTZ")
    INTERVAL = PostgresSqlTypeDef("INTERVAL")

    # JSON types
    JSON = PostgresSqlTypeDef("JSON")
    JSONB = PostgresSqlTypeDef("JSONB")

    # Other types
    ARRAY = PostgresSqlTypeDef("ARRAY")
    UUID = PostgresSqlTypeDef("UUID")
    BYTEA = PostgresSqlTypeDef("BYTEA")

    VECTOR = PostgresSqlTypeDef("vector")  # Generic vector type

    @staticmethod
    def vector(dimensions: int) -> PostgresSqlTypeDef:
        """Create a vector type with specific dimensions for pgvector."""
        return PostgresSqlTypeDef("vector", dimensions=dimensions)

    # ============================================================================
    # pgvector Index Factories
    # ============================================================================

    @staticmethod
    def ivfflat_cosine(table: str, column: str, lists: int = 100) -> PostgresSqlVectorTypeDef:
        """
        Create an IVFFlat index using cosine distance for vector similarity search.

        IVFFlat is faster but less accurate than HNSW. Good for large datasets.

        Args:
            table: Table name
            column: Column name containing vectors
            lists: Number of inverted lists (default 100, recommended: rows/1000)

        Returns:
            PostgresSqlVectorTypeDef instance for creating the index
        """
        return PostgresSqlVectorTypeDef(
            "ivfflat_cosine",
            idx_table=table,
            idx_column=column,
            idx_lists=lists,
            index_type=PGVectorIndexes.IVFFLAT_COSINE
        )

    @staticmethod
    def ivfflat_l2(table: str, column: str, lists: int = 100) -> PostgresSqlVectorTypeDef:
        """
        Create an IVFFlat index using L2 (Euclidean) distance.

        Args:
            table: Table name
            column: Column name containing vectors
            lists: Number of inverted lists (default 100)

        Returns:
            PostgresSqlVectorTypeDef instance for creating the index
        """
        return PostgresSqlVectorTypeDef(
            "ivfflat_l2",
            idx_table=table,
            idx_column=column,
            idx_lists=lists,
            index_type=PGVectorIndexes.IVFFLAT_L2
        )

    @staticmethod
    def ivfflat_ip(table: str, column: str, lists: int = 100) -> PostgresSqlVectorTypeDef:
        """
        Create an IVFFlat index using inner product distance.

        Args:
            table: Table name
            column: Column name containing vectors
            lists: Number of inverted lists (default 100)

        Returns:
            PostgresSqlVectorTypeDef instance for creating the index
        """
        return PostgresSqlVectorTypeDef(
            "ivfflat_ip",
            idx_table=table,
            idx_column=column,
            idx_lists=lists,
            index_type=PGVectorIndexes.IVFFLAT_IP
        )

    @staticmethod
    def hnsw_cosine(table: str, column: str, m: int = 16, ef_construction: int = 64) -> PostgresSqlVectorTypeDef:
        """
        Create an HNSW index using cosine distance.

        HNSW is more accurate but slower to build than IVFFlat. Better for smaller datasets.

        Args:
            table: Table name
            column: Column name containing vectors
            m: Max number of connections per layer (default 16, higher = better recall, more memory)
            ef_construction: Size of dynamic candidate list for construction (default 64, higher = better quality)

        Returns:
            PostgresSqlVectorTypeDef instance for creating the index
        """
        return PostgresSqlVectorTypeDef(
            "hnsw_cosine",
            idx_table=table,
            idx_column=column,
            idx_m=m,
            idx_ef_construction=ef_construction,
            index_type=PGVectorIndexes.HNSW_COSINE
        )

    @staticmethod
    def hnsw_l2(table: str, column: str, m: int = 16, ef_construction: int = 64) -> PostgresSqlVectorTypeDef:
        """
        Create an HNSW index using L2 (Euclidean) distance.

        Args:
            table: Table name
            column: Column name containing vectors
            m: Max number of connections per layer (default 16)
            ef_construction: Size of dynamic candidate list for construction (default 64)

        Returns:
            PostgresSqlVectorTypeDef instance for creating the index
        """
        return PostgresSqlVectorTypeDef(
            "hnsw_l2",
            idx_table=table,
            idx_column=column,
            idx_m=m,
            idx_ef_construction=ef_construction,
            index_type=PGVectorIndexes.HNSW_L2
        )

    @staticmethod
    def hnsw_ip(table: str, column: str, m: int = 16, ef_construction: int = 64) -> PostgresSqlVectorTypeDef:
        """
        Create an HNSW index using inner product distance.

        Args:
            table: Table name
            column: Column name containing vectors
            m: Max number of connections per layer (default 16)
            ef_construction: Size of dynamic candidate list for construction (default 64)

        Returns:
            PostgresSqlVectorTypeDef instance for creating the index
        """
        return PostgresSqlVectorTypeDef(
            "hnsw_ip",
            idx_table=table,
            idx_column=column,
            idx_m=m,
            idx_ef_construction=ef_construction,
            index_type=PGVectorIndexes.HNSW_IP
        )

    # @staticmethod
    # def ivfflat_inner_product(table: str, column: str, lists: int = 100) -> str:
    #     """Generate SQL for an IVFFLAT index using inner product similarity."""
    #     return (
    #         f"CREATE INDEX {table}_{column}_ivfflat_ip_idx "
    #         f"ON {table} USING ivfflat ({column} vector_ip_ops) "
    #         f"WITH (lists = {lists});"
    #     )


class PostgresSqlConnector(SqlConnector):
    def __init__(self, dbConfig):
        super().__init__(dbConfig)

        self.schema = dbConfig.get("schema", "public")  # Default to 'public' if schema is not provided

    def _connect_to_db(self):
        """Connects to the PostgreSQL database."""
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(
                dbname=self._config["database"],
                user=self._config["user"],
                password=self._config["password"],
                host=self._config["host"],
                port=self._config["port"],
                options=self._config.get("options", f"-c search_path={self.schema}")
            )
            cur = conn.cursor()
            return conn, cur
        except Exception as e:
            Utility.error_log(f"Failed to connect to the database: {e}")

        return conn, cur

    def _disconnect_from_db(self, conn, cur):
        """Disconnects from the PostgreSQL database."""
        if cur:
            cur.close()
        if conn:
            conn.close()

    def exec_sql(self, query):
        """Executes a single SQL command."""
        conn, cur = self._connect_to_db()
        try:
            cur.execute(query)
            conn.commit()
        except Exception as e:
            Utility.error_log(f"Error executing SQL: {e}")
            conn.rollback()
        finally:
            self._disconnect_from_db(conn, cur)

    def create_table_using_data_class(self, data_class: Type, table_name: str = None):
        """Creates a table in the database based on the provided data class."""
        if table_name is None:
            table_name = data_class.__name__.lower()  # Table name based on class name

        # Build columns definition, primary key detection, and foreign key constraints
        columns = []
        pkey_columns = []
        fkey_constraints = []

        for field in fields(data_class):
            # Check if the field is a primary key and set the appropriate type
            if field.metadata.get("is_pkey"):
                column_definition = f"{field.name} SERIAL PRIMARY KEY"  # Use SERIAL for auto-increment
                pkey_columns.append(field.name)
            elif field.metadata.get("auto_time_stamp"):
                column_definition = f"{field.name} timestamp default current_timestamp"
            else:
                column_definition = f"{field.name} {self._get_sql_type(field.type)}"

            # Check for foreign key and resolve referenced table/column
            if field.metadata.get("is_fkey"):
                references_class = field.metadata["references_class"]  # This is already stored as a string
                references_column = field.metadata["references_column"]  # This is a string of the column name
                fkey_constraints.append(
                    f", FOREIGN KEY ({field.name}) REFERENCES {self.schema}.{references_class}({references_column})"
                )

            columns.append(column_definition)

        # Join all column definitions and foreign key constraints
        columns_definition = ', '.join(columns) + ''.join(fkey_constraints)

        # Use 'DROP TABLE IF EXISTS' to safely drop the table if it exists
        query = f"DROP TABLE IF EXISTS {self.schema}.{table_name};"
        self.exec_sql(query)

        query = f"CREATE TABLE IF NOT EXISTS {self.schema}.{table_name} ({columns_definition});"
        self.exec_sql(query)

    def _get_sql_type(self, python_type) -> PostgresSqlTypeDef:
        """
        Maps Python types to PostgreSQL types using PostgresSqlTypeDef dataclass.
        Supports pgvector extension for vector/embedding types.

        Args:
            python_type: Python type or type annotation to map

        Returns:
            PostgresSqlTypeDef: PostgreSQL type instance

        Examples:
            >>> _get_sql_type(int)
            PostgresSqlTypeDef(base_type='INTEGER')
            >>> _get_sql_type(dict)
            PostgresSqlTypeDef(base_type='JSONB')
            >>> _get_sql_type(list)
            PostgresSqlTypeDef(base_type='JSONB')  # Arrays/lists default to JSONB for flexibility
        """
        # Mapping from Python types to PostgreSQL types
        type_mapping = {
            int: PostgresDataType.INTEGER,
            str: PostgresDataType.VARCHAR,
            float: PostgresDataType.FLOAT,
            bool: PostgresDataType.BOOLEAN,
            dict: PostgresDataType.JSONB,
            list: PostgresDataType.JSONB,  # Lists default to JSONB (can be ARRAY or vector depending on use case)
            datetime: PostgresDataType.TIMESTAMP
        }

        # Try direct type lookup first
        if python_type in type_mapping:
            return type_mapping[python_type]

        # Handle type annotations (Optional, Union, List, etc.)
        type_str = str(python_type).lower()

        # String-based type mapping for annotations
        string_type_mapping = {
            'int': PostgresDataType.INTEGER,
            'str': PostgresDataType.VARCHAR,
            'float': PostgresDataType.FLOAT,
            'bool': PostgresDataType.BOOLEAN,
            'dict': PostgresDataType.JSONB,
            'list': PostgresDataType.JSONB,
            'datetime': PostgresDataType.TIMESTAMP,
            'date': PostgresDataType.DATE,
            'time': PostgresDataType.TIME,
            'json': PostgresDataType.JSONB,
            'jsonb': PostgresDataType.JSONB,
            'vector': PostgresDataType.VECTOR  # Generic vector without dimensions
        }

        # Check for type hints in string representation
        if 'optional' in type_str or 'union' in type_str:
            for type_name, pg_type in string_type_mapping.items():
                if type_name in type_str:
                    return pg_type

        # Direct string matching
        for type_name, pg_type in string_type_mapping.items():
            if type_name in type_str:
                return pg_type

        # Default to VARCHAR for unknown types
        return PostgresDataType.VARCHAR

    def execute_sql_file(self, file_path: str):
        """Executes SQL commands from a file.

        Args:
            file_path (str): The path to the SQL file containing commands to be executed.
        """
        try:
            with open(file_path, 'r') as file:
                sql_commands = file.read()
                # Execute the entire content of the SQL file as one command
                self.exec_sql(sql_commands)
            Utility.log(f"Executed SQL commands from {file_path}.")
        except Exception as e:
            Utility.error_log(f"Error executing SQL file {file_path}: {e}")

    def create_table_using_def(self, table_name: str, table_def: dict):
        """Creates a table using the provided definition.

        Args:
            table_name (str): The name of the table to be created.
            table_def (dict): A dictionary where keys are column names and values are data types.
        """
        columns = ', '.join([f"{col} {dtype}" for col, dtype in table_def.items()])
        query = f"CREATE TABLE IF NOT EXISTS {self.schema}.{table_name} ({columns});"
        self.exec_sql(query)

    def bulk_insert_data_into_table(self, table_name: str, df: pd.DataFrame):
        """Inserts a DataFrame's data into the specified table.

        Args:
            table_name (str): The name of the table.
            df (pd.DataFrame): The DataFrame containing the data to insert.
        """
        if df.empty:
            Utility.warning_log("DataFrame is empty; nothing to insert.")
            return

        # Convert DataFrame values to JSON for any dictionary-like entries
        def convert_to_json(value):
            if isinstance(value, dict):
                return json.dumps(value)  # Convert dict to JSON string
            return value

        # Apply the conversion function to all elements in the DataFrame
        df = df.applymap(convert_to_json)

        cols = list(df.columns)
        query = f"INSERT INTO {self.schema}.{table_name} ({', '.join(cols)}) VALUES %s"

        # Using execute_values for efficient bulk inserts
        conn, cur = self._connect_to_db()
        try:
            execute_values(cur, query, df.values)
            conn.commit()
        except Exception as e:
            Utility.error_log(f"Error inserting data: {e}")
            conn.rollback()
        finally:
            self._disconnect_from_db(conn, cur)

    def run_sql_and_get_df(self, query, warnings: bool = False) -> pd.DataFrame:
        """Runs a SQL query and returns the results as a DataFrame.

        Args:
            query (str): The SQL query to run.
            warnings (bool): Whether to display warnings (default: False).

        Returns:
            pd.DataFrame: The resulting DataFrame.
        """
        conn, cur = self._connect_to_db()
        try:
            cur.execute(query)
            data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            Utility.error_log(f"Error running query: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error
        finally:
            self._disconnect_from_db(conn, cur)

    def pull_data_from_table(self, table_name: str, columns: [str], filters: dict = None) -> pd.DataFrame:
        cols_sep = ",".join(columns)

        # Build the filters into the WHERE clause, if any
        filter_sql = ""
        if filters:
            filter_conditions = []
            for column, condition in filters.items():
                # Handle basic condition as a direct value, defaults to '=' operator
                if not isinstance(condition, tuple):
                    # If condition is a string, handle quotes
                    if isinstance(condition, str):
                        filter_conditions.append(f"{column} = '{condition}'")
                    elif isinstance(condition, list) or isinstance(condition, tuple):  # For IN operator
                        values = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in condition])
                        filter_conditions.append(f"{column} IN ({values})")
                    else:
                        filter_conditions.append(f"{column} = {condition}")
                else:
                    # Handle operator in the condition tuple: (operator, value)
                    operator, value = condition
                    if isinstance(value, str):
                        if operator.upper() == 'LIKE':
                            filter_conditions.append(f"{column} {operator} '%{value}%'")
                        else:
                            filter_conditions.append(f"{column} {operator} '{value}'")
                    elif isinstance(value, list) or isinstance(value, tuple):
                        values = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])
                        filter_conditions.append(f"{column} {operator} ({values})")
                    else:
                        filter_conditions.append(f"{column} {operator} {value}")

            filter_sql = " WHERE " + " AND ".join(filter_conditions)

        # Build the final SQL query
        query = f"""
                SELECT {cols_sep}
                FROM {self.schema}.{table_name}
                {filter_sql}
            """

        Utility.log(f"running:\n {query}")

        df = self.run_sql_and_get_df(query)

        return df

    def get_table(self, table_name, limit_n: int = None):
        """Retrieves data from a specified table as a DataFrame.

        Args:
            table_name (str): The name of the table.
            limit_n (int, optional): Maximum number of rows to retrieve.

        Returns:
            pd.DataFrame: The resulting DataFrame containing the rows.
        """
        limit_clause = f"LIMIT {limit_n}" if limit_n else ""
        query = f"SELECT * FROM {self.schema}.{table_name} {limit_clause};"
        return self.run_sql_and_get_df(query)

    def get_table_n_rows_to_df(self, tableName: str, nrows: int) -> pd.DataFrame:
        """Retrieves the first n rows from a specified table as a DataFrame.

        Args:
            tableName (str): The name of the table.
            nrows (int): The number of rows to retrieve.

        Returns:
            pd.DataFrame: The resulting DataFrame containing the rows.
        """
        query = f"SELECT * FROM {self.schema}.{tableName} LIMIT {nrows};"
        return self.run_sql_and_get_df(query)

    def drop_table(self, tableName: str):
        """Drops a specified table if it exists.

        Args:
            tableName (str): The name of the table to drop.
        """
        query = f"DROP TABLE IF EXISTS {self.schema}.{tableName};"
        self.exec_sql(query)

    def list_tables(self) -> List[str]:
        """Lists all tables in the current schema.

        Returns:
            List[str]: List of table names in the schema.
        """
        query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.schema}'
        ORDER BY table_name;
        """
        result = self.run_sql_and_get_df(query)
        return result['table_name'].tolist() if not result.empty else []

    def list_tables_with_prefix(self, prefix: str) -> List[str]:
        """Lists all tables in the current schema that start with the given prefix.

        Args:
            prefix (str): The prefix to filter table names.

        Returns:
            List[str]: List of table names matching the prefix.
        """
        query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = '{self.schema}'
        AND table_name LIKE '{prefix}%'
        ORDER BY table_name;
        """
        result = self.run_sql_and_get_df(query)
        return result['table_name'].tolist() if not result.empty else []

    def create_table_ctas(self, tableName: str, innerSql: str, sortkey: str = None, distkey: str = None,
                          include_print: bool = True):
        """Creates a table using a CREATE TABLE AS SELECT (CTAS) statement.

        Args:
            tableName (str): The name of the new table.
            innerSql (str): The inner SQL query to use for creating the table.
            sortkey (str, optional): Sort key for the new table.
            distkey (str, optional): Distribution key for the new table.
            include_print (bool): Whether to print a confirmation message (default: True).
        """
        query = f"CREATE TABLE {self.schema}.{tableName} AS {innerSql};"
        self.exec_sql(query)
        if include_print:
            Utility.log(f"Table {self.schema}.{tableName} created using CTAS.")

    def append_to_table_insert_select(self, tableName: str, innerSql: str, columnStr: str = None):
        """Appends rows to a specified table using an INSERT ... SELECT statement.

        Args:
            tableName (str): The name of the table to append to.
            innerSql (str): The inner SQL query providing the rows to insert.
            columnStr (str, optional): Column names for the INSERT statement.
        """
        query = f"INSERT INTO {self.schema}.{tableName} {columnStr} SELECT * FROM ({innerSql}) as subquery;"
        self.exec_sql(query)

    def get_table_header(self, tableName: str) -> [str]:
        """Retrieves the column names of a specified table.

        Args:
            tableName (str): The name of the table.

        Returns:
            List[str]: The list of column names.
        """
        query = f"SELECT * FROM {tableName} LIMIT 0;"
        self.cur.execute(query)
        return [desc[0] for desc in self.cur.description]

    def clone_table(self, tableName: str, newTableName: str):
        """Clones a specified table.

        Args:
            tableName (str): The name of the table to clone.
            newTableName (str): The name for the new cloned table.
        """
        query = f"CREATE TABLE {newTableName} (LIKE {tableName} INCLUDING ALL);"
        self.exec_sql(query)

    def rename_table(self, table_name: str, new_table_name: str):
        """Renames a specified table.

        Args:
            table_name (str): The current name of the table.
            new_table_name (str): The new name for the table.
        """
        query = f"ALTER TABLE {self.schema}.{table_name} RENAME TO {new_table_name};"
        self.exec_sql(query)

    def check_if_table_exists(self, table_name: str) -> bool:
        """Checks if a specified table exists.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}');"
        result = self.run_sql_and_get_df(query)
        return result.iloc[0, 0]

    def get_table_row_count(self, table_name: str, warnings: bool = False) -> int:
        """Gets the row count of a specified table.

        Args:
            table_name (str): The name of the table.
            warnings (bool): Whether to display warnings (default: False).

        Returns:
            int: The number of rows in the table.
        """
        query = f"SELECT COUNT(*) FROM {self.schema}.{table_name};"
        result = self.run_sql_and_get_df(query)
        return result.iloc[0, 0]

    def get_distinct_values_from_single_col(self, column_name: str, table_name: str):
        """Retrieves distinct values from a specified column of a table.

        Args:
            column_name (str): The name of the column.
            table_name (str): The name of the table.

        Returns:
            pd.DataFrame: DataFrame containing distinct values from the specified column.
        """
        query = f"SELECT DISTINCT {column_name} FROM {self.schema}.{table_name};"
        return self.run_sql_and_get_df(query)

    def test_table_by_row_count(self, table_name: str):
        """Tests the row count of a specified table and prints the result.

        Args:
            table_name (str): The name of the table.
        """
        count = self.get_table_row_count(table_name)
        Utility.log(f"Row count for {self.schema}.{table_name}: {count}")

    def insert_row_to_table(self, table_name: str, row: dict) -> int:
        """
        Inserts a row into the specified table and returns the newly created ID.

        Args:
            table_name (str): The name of the table to insert the row into.
            row (dict): A dictionary where keys are column names and values are the data to insert.

        Returns:
            int: The ID of the newly created row.
        """
        # Extract columns and values from the row dictionary
        columns = ', '.join(row.keys())

        # Prepare placeholders for parameterized queries
        placeholders = ', '.join(['%s'] * len(row))

        # Construct the SQL query
        query = f"""
        INSERT INTO {self.schema}.{table_name} ({columns}) 
        VALUES ({placeholders}) 
        RETURNING id;
        """

        # Prepare values for the query, converting dicts to JSONB if necessary
        values = []
        for value in row.values():
            if isinstance(value, dict):
                values.append(json.dumps(value))  # Convert dictionary to JSON string for JSONB field
            else:
                values.append(value)

        # Execute the query and fetch the newly inserted ID
        try:
            result = self.run_sql_and_get_one(query,
                                              values)  # Assuming run_sql_and_get_one executes a query and returns a single row
            new_id = result['id']  # Assuming the ID column is named 'id'
            return new_id
        except Exception as e:
            Utility.error_log(f"Error inserting row: {e}")
            return None

    def clear_table(self, tableName: str):
        query = f"DELETE FROM {self.schema}.{tableName};"
        self.exec_sql(query)

    def check_db_status(self) -> bool:
        try:
            conn, cur = self._connect_to_db()
            Utility.log("Database is up and running.")
            return True
        except Exception as e:
            Utility.error_log(f"Database connection failed: {e}")
            return False

    def update_table_structure_using_data_class(self, data_class: Type, table_name: str = None):
        """Updates the table structure to match the provided data class by adding/removing columns."""
        if table_name is None:
            table_name = data_class.__name__.lower()

        # Fetch the existing table columns from the database
        existing_columns = {col.lower(): dtype for col, dtype in self._get_table_columns(table_name).items()}

        # Build the set of new columns from the data class, using lowercase for case-insensitivity
        data_class_columns = {field.name.lower(): self._get_sql_type(field.type) for field in fields(data_class)}

        # Determine columns to add (new columns in the data class, but not in the DB)
        columns_to_add = {name: sql_type for name, sql_type in data_class_columns.items() if
                          name not in existing_columns}

        # Determine columns to remove (existing in DB but not in the data class)
        columns_to_remove = {name for name in existing_columns if name not in data_class_columns}

        # Detect if changes are needed
        if not columns_to_add and not columns_to_remove:
            Utility.log(f"Table {self.schema}.{table_name} is already up-to-date.")
            return

        # Generate the necessary ALTER TABLE statements
        alter_statements = []

        # Add new columns
        for column_name, column_type in columns_to_add.items():
            alter_statements.append(f"ALTER TABLE {self.schema}.{table_name} ADD COLUMN {column_name} {column_type};")

        # Handle foreign key dependencies and prevent dropping critical columns like primary keys
        if columns_to_remove:
            Utility.warning_log(f"Detected deprecated columns... {columns_to_remove}... Columns not allowed to be removed from mode")
            for column_name in columns_to_remove:
                # Skip primary key and foreign key columns to avoid conflicts
                if column_name in [col.lower() for col in self._get_primary_key_columns(table_name)]:
                    Utility.log(f"Skipping removal of primary key column: {column_name}")
                    continue

                fk_constraints = self._get_foreign_key_constraints(table_name, column_name)
                if fk_constraints:
                    Utility.log(f"Skipping removal of foreign key column: {column_name} (FK constraints found)")
                    continue

                # If it's safe to remove, drop the column
                alter_statements.append(f"ALTER TABLE {self.schema}.{table_name} DROP COLUMN {column_name};")

        # Execute the ALTER TABLE statements
        for statement in alter_statements:
            try:
                self.exec_sql(statement)
            except Exception as e:
                Utility.log(f"Error executing SQL: {statement}. Error: {str(e)}")

        Utility.log(f"Table {self.schema}.{table_name} has been updated successfully.")

    def _get_table_columns(self, table_name: str) -> Dict[str, str]:
        """Fetches the current column names and types from the database for the given table."""
        query = f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = '{self.schema}' AND table_name = '{table_name}';
        """
        result = self.run_sql_and_get_df(query)
        return {row['column_name']: row['data_type'] for _, row in result.iterrows()}

    def _get_foreign_key_constraints(self, table_name: str, column_name: str) -> List[str]:
        """Retrieve the foreign key constraints for the specified table and column."""
        query = f"""
        SELECT constraint_name
        FROM information_schema.key_column_usage
        WHERE table_name = '{table_name}' AND column_name = '{column_name}';
        """
        result = self.run_sql_and_get_df(query)
        return result['constraint_name'].tolist() if not result.empty else []

    def _get_primary_key_columns(self, table_name: str) -> List[str]:
        """Retrieve the primary key columns for the specified table."""
        query = f"""
        SELECT a.attname
        FROM   pg_index i
        JOIN   pg_attribute a ON a.attrelid = i.indrelid
                             AND a.attnum = ANY(i.indkey)
        WHERE  i.indrelid = '{table_name}'::regclass
        AND    i.indisprimary;
        """
        result = self.run_sql_and_get_df(query)
        return result['attname'].tolist() if not result.empty else []

    def insert_data_model_to_table(self, data_model: Any, table_name: str, pk_key: str = "ID") -> int:
        """Inserts a data model into the specified table and returns the ID of the inserted row.

        Args:
            data_model (Type): The data model instance to insert.
            table_name (str): The name of the table.
            pk_key (str): The primary key field to exclude from insert. Defaults to "ID".

        Returns:
            int: The ID of the inserted row.
        """
        # Convert the dataclass instance to a dictionary
        data_dict = asdict(data_model)

        # Exclude the primary key from the data_dict if it exists
        data_dict.pop(pk_key, None)  # Remove the primary key field

        # Convert dicts to JSON strings, since PostgreSQL can store JSON data types
        for key, value in data_dict.items():
            if isinstance(value, dict):
                data_dict[key] = psycopg2.extras.Json(value) #json.dumps(value)

        # Generate the SQL insert statement (excluding ID)
        columns = ', '.join(data_dict.keys())
        values = ', '.join(['%s'] * len(data_dict))
        query = f"INSERT INTO {self.schema}.{table_name} ({columns}) VALUES ({values}) RETURNING id;"

        conn, cur = self._connect_to_db()
        try:
            cur.execute(query, tuple(data_dict.values()))
            conn.commit()
            return cur.fetchone()[0]  # Return the inserted row's ID
        except Exception as e:
            Utility.error_log(f"Error inserting data: {e}")
            conn.rollback()
            return -1  # Indicate failure
        finally:
            self._disconnect_from_db(conn, cur)

    def update_data_model_to_table(self, data_model: Type, table_name: str, identifier_column: str) -> bool:
        """Inserts or updates a record in the specified table based on a unique identifier.

        Args:
            data (Type): The data model instance.
            table_name (str): The name of the table.
            identifier_column (str): The unique identifier column to find the record.

        Returns:
            bool: True if the operation was successful, False otherwise.
            :param data_model:
        """
        data_dict = asdict(data_model)

        identifier_col = data_model.get_field_name(identifier_column)

        if data_dict[identifier_col] is None:
            data_dict.pop(identifier_column, None)  # Remove the primary key field
        else:
            data_dict[identifier_column] = data_dict[identifier_col]
            if identifier_column not in data_dict:
                Utility.error_log(f"Error: Identifier column '{identifier_column}' not found in data.")
                return False

        columns = ', '.join(data_dict.keys())
        values = ', '.join(['%s'] * len(data_dict))
        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in data_dict.keys() if col != identifier_column])

        # Construct the query
        query = f"""
            INSERT INTO {self.schema}.{table_name} ({columns}) 
            VALUES ({values}) 
            ON CONFLICT ({identifier_column}) 
            DO UPDATE SET {update_clause};
        """

        conn, cur = self._connect_to_db()
        try:
            # Execute the query with safe parameters
            cur.execute(query, tuple(data_dict.values()))
            conn.commit()
            return True  # Indicate success
        except Exception as e:
            Utility.error_log(f"Error during upsert: {e}")
            conn.rollback()
            return False
        finally:
            self._disconnect_from_db(conn, cur)

    def upsert_data_model_to_table(self, data_model: Type, table_name: str, identifier_column: str) -> bool:
        """Inserts or updates a record in the specified table based on a unique identifier.

        Args:
            data (Type): The data model instance.
            table_name (str): The name of the table.
            identifier_column (str): The unique identifier column to find the record.

        Returns:
            bool: True if the operation was successful, False otherwise.
            :param data_model:
        """
        data_dict = asdict(data_model)

        # Convert dicts to JSON strings, since PostgreSQL can store JSON data types
        for key, value in data_dict.items():
            if isinstance(value, dict):
                data_dict[key] = psycopg2.extras.Json(value)

        identifier_col = data_model.get_field_name(identifier_column)

        if data_dict[identifier_col] is None:
            data_dict.pop(identifier_column, None)  # Remove the primary key field
        else:
            data_dict[identifier_column] = data_dict[identifier_col]
            if identifier_column not in data_dict:
                Utility.error_log(f"Error: Identifier column '{identifier_column}' not found in data.")
                return False

        columns = ', '.join(data_dict.keys())
        values = ', '.join(['%s'] * len(data_dict))
        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in data_dict.keys() if col != identifier_column])
        query = f"""
            INSERT INTO {self.schema}.{table_name} ({columns}) 
            VALUES ({values}) 
            ON CONFLICT ({identifier_column}) 
            DO UPDATE SET {update_clause};
        """

        conn, cur = self._connect_to_db()
        try:
            cur.execute(query, tuple(data_dict.values()))
            conn.commit()
            return True  # Indicate success
        except Exception as e:
            Utility.error_log(f"Error during upsert: {e}")
            conn.rollback()
            return False
        finally:
            self._disconnect_from_db(conn, cur)

    def get_data_model_from_db(self, data_class: Type, filters: Dict[str, Any], table_name: str = None) -> List[Any]:
        """Retrieve data from the database based on the provided data model and filters.

        Args:
            data_model (Type): The data class representing the table structure.
            filters (Dict[str, Any]): A dictionary of filters to apply in the query.

        Returns:
            List[Any]: A list of instances of the data class containing the results of the query.
            :param filters:
            :param data_class:
            :param table_name:
        """
        if table_name is None:
            table_name = data_class.__name__.lower()  # Convert class name to table name
        filter_conditions = []

        # Build filter conditions based on the filters provided
        for field in fields(data_class):
            if field.name in filters:
                # Directly add the condition to the list
                filter_conditions.append(f"{field.name} = '{filters[field.name]}'")  # Use string interpolation

        # Construct the SQL query
        if filter_conditions:
            filter_query = " AND ".join(filter_conditions)
            query = f"SELECT * FROM {self.schema}.{table_name} WHERE {filter_query};"
        else:
            query = f"SELECT * FROM {self.schema}.{table_name};"

        # Execute the query and get results
        results = self.run_sql_and_get_df(query, warnings=False)

        # Map results to data class instances
        data_instances = [
            data_class(**dict(zip([field.name for field in fields(data_class)], row)))
            for row in results.values.tolist()  # Convert DataFrame to list of rows
        ]

        return data_instances

    def write_dataframe_to_table(self, df, table_name: str, mode: str = "append"):
        """Writes a pandas DataFrame to a database table.

        Args:
            df: Pandas DataFrame to write
            table_name (str): Target table name
            mode (str): Write mode - 'append', 'replace', 'fail'
                - 'append': Insert new records (default)
                - 'replace': Drop table and recreate
                - 'fail': Raise error if table exists

        Note:
            This method is designed for pandas DataFrames.
            For Spark DataFrames, use SparkSQLConnector instead.
        """
        from sqlalchemy import create_engine

        # Create SQLAlchemy connection string
        connection_string = f"postgresql://{self._config['user']}:{self._config['password']}@{self._config['host']}:{self._config['port']}/{self._config['dbname']}"

        # Add options if they exist
        if 'options' in self._config and self._config['options']:
            connection_string += f"?options={self._config['options']}"

        try:
            # Create engine
            engine = create_engine(connection_string)

            # Map mode parameter to pandas if_exists parameter
            if_exists_map = {
                'append': 'append',
                'replace': 'replace',
                'overwrite': 'replace',
                'fail': 'fail',
                'error': 'fail'
            }
            if_exists = if_exists_map.get(mode, 'append')

            # Write DataFrame to database
            df.to_sql(
                name=table_name,
                con=engine,
                schema=self.schema,
                if_exists=if_exists,
                index=False,
                method='multi'  # Use multi-row INSERT for better performance
            )

            Utility.log(f"Successfully wrote {len(df)} rows to {self.schema}.{table_name} (mode: {mode})")

        except Exception as e:
            Utility.error_log(f"Error writing DataFrame to table {table_name}: {e}")
            raise
        finally:
            if 'engine' in locals():
                engine.dispose()

    def cast_column(self, table_name: str, column_name: str, type: PostgresSqlTypeDef):
        """
        Cast a column to a different type using ALTER TABLE.

        Args:
            table_name: Name of the table
            column_name: Name of the column to cast
            type: PostgresSqlTypeDef instance representing the target type

        Example:
            >>> connector.cast_column("users", "age", INTEGER)
            >>> connector.cast_column("products", "embedding", PostgresDataType.vector(1536))
        """
        query = f"ALTER TABLE {self.schema}.{table_name} ALTER COLUMN {column_name} TYPE {type.to_sql()};"
        self.exec_sql(query)

    def create_index_column(self, index: PostgresSqlVectorTypeDef):
        """
        Create an index on a specific column for a given table.

        The index object contains all necessary information (table, column, type, parameters).
        Supports pgvector indexes (ivfflat with cosine/L2 distance operators).

        Args:
            index: PostgresSqlVectorTypeDef instance containing index configuration

        Example:
            >>> # Create ivfflat index with cosine similarity
            >>> index = PostgresDataType.ivfflat_cosine('products', 'embedding', lists=100)
            >>> connector.create_index_column(index)
        """
        index_sql = index.to_sql()
        self.exec_sql(index_sql)

