from __future__ import annotations

from typing import Type, Dict, Any, List, Optional
from dataclasses import dataclass

import pandas as pd
from pyspark.sql import DataFrame

from abc import abstractmethod, ABC


@dataclass(frozen=True)
class SqlType:
    """
    Base class for SQL data types across different database connectors.

    Provides a common interface for representing SQL types with optional parameters.
    Subclasses should implement database-specific type systems (e.g., PostgresSqlType, MySqlType).

    Immutable (frozen) to allow use in sets/dicts and ensure consistency.

    Attributes:
        base_type: The base SQL type name (e.g., 'INTEGER', 'VARCHAR', 'vector')
        length: Optional length parameter (e.g., VARCHAR(255))
        precision: Optional precision parameter (e.g., NUMERIC(10,2))
        scale: Optional scale parameter (e.g., NUMERIC(10,2))
        dimensions: Optional dimensions parameter (e.g., vector(1536) for pgvector)
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

    def __str__(self) -> str:
        """String representation returns SQL type"""
        return self.to_sql()


class SqlConnector(ABC):
    def __init__(self, dbConfig):
        self._config = dbConfig

    @abstractmethod
    def _connect_to_db(self):
        pass

    @abstractmethod
    def _disconnect_from_db(self, conn, cur):
        pass

    @abstractmethod
    def execute_sql_file(self, file_path: str):
        pass

    @abstractmethod
    def create_table_using_def(self, table_name: str, table_def: dict):
        pass

    @abstractmethod
    def create_table_using_data_class(self, data_class: Type, table_name: str = None):
        pass

    @abstractmethod
    def update_table_structure_using_data_class(self, data_class: Type, table_name: str = None):
        pass

    @abstractmethod
    def bulk_insert_data_into_table(self, table_name: str, df: pd.DataFrame()):
        pass

    @abstractmethod
    def get_table(self, table_name, limit_n: int = None):
        pass

    @abstractmethod
    def run_sql_and_get_df(self, query, warnings: bool) :
        pass

    @abstractmethod
    def pull_data_from_table(self, table_name: str, columns: [str], filters: dict = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def exec_sql(self, query):
        pass

    @abstractmethod
    def get_table_n_rows_to_df(self, tableName: str, nrows: int):
        pass

    @abstractmethod
    def drop_table(self, tableName: str):
        pass

    @abstractmethod
    def list_tables_with_prefix(self, prefix: str) -> List[str]:
        """
        Lists all tables that start with the given prefix.

        :param prefix: The prefix to filter table names
        :return: List of table names matching the prefix
        """
        pass

    @abstractmethod
    def list_tables(self) -> List[str]:
        pass

    @abstractmethod
    def create_table_ctas(self, tableName: str, innerSql: str, sortkey: str = None, distkey: str = None,
                          include_print: bool = True):
        pass

    @abstractmethod
    def append_to_table_insert_select(self, tableName: str, innerSql: str, columnStr: str = None):
        pass

    @abstractmethod
    def get_table_header(self, tableName: str) -> [str]:
        pass

    @abstractmethod
    def clone_table(self, tableName: str, newTableName: str):
        pass

    @abstractmethod
    def rename_table(self, table_name, new_table_name):
        pass

    @abstractmethod
    def check_if_table_exists(self, table_name) -> bool:
        pass

    @abstractmethod
    def get_table_row_count(self, table_name, warnings: bool) -> int:
        pass

    @abstractmethod
    def get_distinct_values_from_single_col(self, column_name: str, table_name: str):
        pass

    @abstractmethod
    def test_table_by_row_count(self, table_name):
        pass

    @abstractmethod
    def clear_table(self, tableName: str):
        pass

    @abstractmethod
    def check_db_status(self) -> bool:
        pass

    @abstractmethod
    def insert_row_to_table(self, table_name: str, row: dict) -> int:
        pass

    @abstractmethod
    def get_data_model_from_db(self, data_model: Type, filters: Dict[str, Any], table_name: str = None) -> List[Any]:
        pass

    @abstractmethod
    def upsert_data_model_to_table(self, data_model: Type, table_name: str, identifier_column: str,
                                   identifier_upsert_value: int = None) -> bool:
        pass

    @abstractmethod
    def update_data_model_to_table(self, data: Type, table_name: str, identifier_column: str) -> bool:
        pass

    @abstractmethod
    def insert_data_model_to_table(self, data_class: Any, table_name: str, pk_key: str = "ID") -> int:
        pass

    @abstractmethod
    def write_dataframe_to_table(self, df: DataFrame, table_name: str, mode: str = "append"):
        pass

    @abstractmethod
    def cast_column(self, table_name: str, column_name: str, type: SqlType):
        """
        Cast a column to a different SQL type.

        Args:
            table_name: Name of the table
            column_name: Name of the column to cast
            type: SQL type instance (e.g., PostgresSqlType for PostgreSQL connectors)
        """
        pass
