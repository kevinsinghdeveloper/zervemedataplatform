from typing import Type, Dict, Any, List

import pandas as pd

from abc import abstractmethod, ABC


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
    def run_sql_and_get_df(self, query, warnings: bool) -> pd.DataFrame:
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



