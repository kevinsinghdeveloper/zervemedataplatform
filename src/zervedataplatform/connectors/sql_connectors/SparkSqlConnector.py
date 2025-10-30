from typing import Type, Dict, Any, List
from dataclasses import fields
from pyspark.sql import SparkSession, DataFrame
from zervedataplatform.abstractions.connectors.SqlConnector import SqlConnector
from zervedataplatform.connectors.sql_connectors.SqlConnectorHandler import SqlConnectorHandler


class SparkSQLConnector(SqlConnector):

    def __init__(self, db_config: dict):
        """
        Initialize SparkSQLConnector with a dedicated Spark session for SQL operations.

        :param db_config: Dictionary containing SQL connection details and Spark settings.
        """
        super().__init__(db_config)

        # get sql connector for db config
        self.__helper_db_connector = SqlConnectorHandler.get_sql_connector(db_config)

        # Create Spark session specifically for SQL operations
        self._spark = SparkSession.builder.appName("SparkSQLSession").getOrCreate()

        # Database connection details
        self.db_url = f"jdbc:postgresql://{db_config['host']}:{db_config['port']}/{db_config['database']}"
        self.user = db_config["user"]
        self.password = db_config["password"]
        self.schema = db_config.get("schema", "public")  # Default schema

    def get_spark(self):
        """Returns the Spark session for SQL operations."""
        return self._spark

    def _connect_to_db(self):
        """In Spark, JDBC does not require an explicit connection."""
        return None, None  # Spark handles connections automatically

    def _disconnect_from_db(self, conn, cur):
        """No explicit disconnection needed for Spark JDBC."""
        pass

    def execute_sql_file(self, file_path: str):
        """Executes SQL commands from a file."""
        with open(file_path, 'r') as file:
            sql_commands = file.read()
            self.exec_sql(sql_commands)

    def create_table_using_def(self, table_name: str, table_def: dict):
        """Creates a table using the provided column definitions."""
        columns = ', '.join([f"{col} {dtype}" for col, dtype in table_def.items()])
        query = f"CREATE TABLE IF NOT EXISTS {self.schema}.{table_name} ({columns});"
        self.exec_sql(query)

    def create_table_using_data_class(self, data_class: Type, table_name: str = None):
        """Creates a table in the database based on the provided data class."""
        if table_name is None:
            table_name = data_class.__name__.lower()

        # Infer schema from data class
        columns = ', '.join([f"{field.name} {self._get_sql_type(field.type)}" for field in fields(data_class)])

        query = f"CREATE TABLE IF NOT EXISTS {self.schema}.{table_name} ({columns});"
        self.exec_sql(query)

    def update_table_structure_using_data_class(self, data_class: Type, table_name: str = None):
        """Updates table schema based on a data class definition (adds missing columns)."""
        if table_name is None:
            table_name = data_class.__name__.lower()

        existing_columns = self.get_table_header(table_name)
        new_columns = [field.name for field in fields(data_class)]

        columns_to_add = set(new_columns) - set(existing_columns)

        for column in columns_to_add:
            col_type = self._get_sql_type(getattr(data_class, column))
            query = f"ALTER TABLE {self.schema}.{table_name} ADD COLUMN {column} {col_type};"
            self.exec_sql(query)

    def bulk_insert_data_into_table(self, table_name: str, df: DataFrame):
        raise NotImplementedError("Bulk insert is not implemented for SparkSQLConnector. Use write_dataframe_to_table instead.")

    def run_sql_and_get_df(self, query: str, warnings: bool = False) -> DataFrame:
        """Runs a SQL query and returns a Spark DataFrame."""
        return self.get_spark().read \
            .format("jdbc") \
            .option("url", self.db_url) \
            .option("query", query) \
            .option("user", self.user) \
            .option("password", self.password) \
            .option("driver", "org.postgresql.Driver") \
            .load()

    def get_table(self, table_name, limit_n: int = None) -> DataFrame:
        limit = ""
        if limit_n:
            limit = f"LIMIT {limit_n}"

        return self.run_sql_and_get_df(query=f"SELECT * FROM {self.schema}.{table_name} {limit}")

    def pull_data_from_table(self, table_name: str, columns: List[str], filters: dict = None) -> DataFrame:
        """Fetches data from a table with optional filters."""
        col_str = ', '.join(columns)
        query = f"SELECT {col_str} FROM {self.schema}.{table_name}"

        if filters:
            conditions = " AND ".join([f"{col} = '{val}'" for col, val in filters.items()])
            query += f" WHERE {conditions}"

        return self.run_sql_and_get_df(query)

    def exec_sql(self, query: str):
        """
        Executes a SQL query on the database.

        Uses the helper connector for non-Spark operations (DDL, DML, etc.)
        that Spark doesn't handle well.
        """
        # Delegate to the helper connector (e.g., PostgresSqlConnector)
        # This avoids direct psycopg2 usage and leverages the existing connector
        self.__helper_db_connector.exec_sql(query)

    def get_table_n_rows_to_df(self, table_name: str, nrows: int) -> DataFrame:
        """Fetches a limited number of rows from a table."""
        query = f"SELECT * FROM {self.schema}.{table_name} LIMIT {nrows};"
        return self.run_sql_and_get_df(query)

    def drop_table(self, table_name: str):
        """Drops a table if it exists."""
        query = f"DROP TABLE IF EXISTS {self.schema}.{table_name};"
        self.exec_sql(query)

    def list_tables_with_prefix(self, prefix: str) -> List[str]:
        """
        Lists all tables in the schema that start with the given prefix.

        :param prefix: The prefix to filter table names
        :return: List of table names matching the prefix
        """
        # Note: Remove semicolon and extra whitespace for JDBC compatibility
        query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{self.schema}' AND table_name LIKE '{prefix}%' ORDER BY table_name"

        try:
            df = self.run_sql_and_get_df(query)
            return df.select("table_name").rdd.flatMap(lambda x: x).collect()
        except Exception as e:
            # If query fails, return empty list
            print(f"Error listing tables with prefix '{prefix}': {e}")
            return []

    def list_tables(self) -> List[str]:
        query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{self.schema}' ORDER BY table_name"

        try:
            df = self.run_sql_and_get_df(query)
            return df.select("table_name").rdd.flatMap(lambda x: x).collect()
        except Exception as e:
            # If query fails, return empty list
            print(f"Error listing tables: {e}")
            return []

    def get_table_header(self, table_name: str) -> List[str]:
        """Retrieves column names from a table."""
        query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}';"
        df = self.run_sql_and_get_df(query)
        return df.select("column_name").rdd.flatMap(lambda x: x).collect()

    def check_if_table_exists(self, table_name: str) -> bool:
        """Checks if a table exists."""
        query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}');"
        df = self.run_sql_and_get_df(query)
        return df.collect()[0][0]

    def get_table_row_count(self, table_name: str, warnings: bool = False) -> int:
        """Gets the row count of a table."""
        query = f"SELECT COUNT(*) FROM {self.schema}.{table_name};"
        df = self.run_sql_and_get_df(query)
        return df.collect()[0][0]

    def get_distinct_values_from_single_col(self, column_name: str, table_name: str) -> List[Any]:
        """Gets distinct values from a column."""
        query = f"SELECT DISTINCT {column_name} FROM {self.schema}.{table_name};"
        df = self.run_sql_and_get_df(query)
        return df.select(column_name).rdd.flatMap(lambda x: x).collect()

    def clear_table(self, table_name: str):
        """Deletes all rows from a table."""
        query = f"DELETE FROM {self.schema}.{table_name};"
        self.exec_sql(query)

    def insert_row_to_table(self, table_name: str, row: Dict[str, Any]) -> int:
        """Inserts a row into a table."""
        columns = ', '.join(row.keys())
        values = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in row.values()])
        query = f"INSERT INTO {self.schema}.{table_name} ({columns}) VALUES ({values}) RETURNING id;"
        df = self.run_sql_and_get_df(query)
        return df.collect()[0][0]

    def get_data_model_from_db(self, data_model: Type, filters: Dict[str, Any], table_name: str = None) -> List[Any]:
        """Fetches data as a list of model instances."""
        df = self.pull_data_from_table(table_name or data_model.__name__.lower(), [field.name for field in fields(data_model)], filters)
        return [data_model(**row.asDict()) for row in df.collect()]

    def _get_sql_type(self, python_type) -> str:
        """
        Maps Python types to SQL types.

        Delegates to the helper connector to leverage its type mapping system,
        which includes support for PostgresDataType enum, JSONB, vector types, etc.
        """
        return self.__helper_db_connector._get_sql_type(python_type)

    def create_table_ctas(self, tableName: str, innerSql: str, sortkey: str = None, distkey: str = None,
                          include_print: bool = True):
        """Creates a table using CREATE TABLE AS SELECT (CTAS)."""
        query = f"CREATE TABLE {self.schema}.{tableName} AS {innerSql};"
        if include_print:
            print(f"Creating table {tableName} with CTAS")
        self.exec_sql(query)

    def append_to_table_insert_select(self, tableName: str, innerSql: str, columnStr: str = None):
        """Appends data to a table using INSERT INTO ... SELECT."""
        if columnStr:
            query = f"INSERT INTO {self.schema}.{tableName} ({columnStr}) {innerSql};"
        else:
            query = f"INSERT INTO {self.schema}.{tableName} {innerSql};"
        self.exec_sql(query)

    def clone_table(self, tableName: str, newTableName: str):
        """Clones a table structure and data."""
        query = f"CREATE TABLE {self.schema}.{newTableName} AS SELECT * FROM {self.schema}.{tableName};"
        self.exec_sql(query)

    def rename_table(self, table_name, new_table_name):
        """Renames a table."""
        query = f"ALTER TABLE {self.schema}.{table_name} RENAME TO {new_table_name};"
        self.exec_sql(query)

    def test_table_by_row_count(self, table_name):
        """Tests if a table has rows."""
        count = self.get_table_row_count(table_name, warnings=False)
        return count > 0

    def check_db_status(self) -> bool:
        """Checks if the database connection is working."""
        try:
            query = "SELECT 1 AS status;"
            df = self.run_sql_and_get_df(query)
            result = df.collect()[0][0]
            return result == 1
        except Exception:
            return False

    def upsert_data_model_to_table(self, data_model: Type, table_name: str, identifier_column: str,
                                   identifier_upsert_value: int = None) -> bool:
        """Upserts a data model instance to a table (INSERT ... ON CONFLICT UPDATE)."""
        if table_name is None:
            table_name = data_model.__class__.__name__.lower()

        # Get field values from data model
        field_dict = {field.name: getattr(data_model, field.name) for field in fields(data_model)}

        columns = ', '.join(field_dict.keys())
        values = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in field_dict.values()])

        # Build update set clause
        update_set = ', '.join([f"{k} = EXCLUDED.{k}" for k in field_dict.keys() if k != identifier_column])

        query = f"""
        INSERT INTO {self.schema}.{table_name} ({columns})
        VALUES ({values})
        ON CONFLICT ({identifier_column})
        DO UPDATE SET {update_set};
        """

        try:
            self.exec_sql(query)
            return True
        except Exception:
            return False

    def update_data_model_to_table(self, data: Type, table_name: str, identifier_column: str) -> bool:
        """Updates a data model in a table."""
        if table_name is None:
            table_name = data.__class__.__name__.lower()

        # Get field values from data model
        field_dict = {field.name: getattr(data, field.name) for field in fields(data)}
        identifier_value = field_dict[identifier_column]

        # Build SET clause
        set_clause = ', '.join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
                               for k, v in field_dict.items() if k != identifier_column])

        # Build WHERE clause with proper value formatting
        if isinstance(identifier_value, str):
            where_clause = f"{identifier_column} = '{identifier_value}'"
        else:
            where_clause = f"{identifier_column} = {identifier_value}"

        query = f"""
        UPDATE {self.schema}.{table_name}
        SET {set_clause}
        WHERE {where_clause};
        """

        try:
            self.exec_sql(query)
            return True
        except Exception:
            return False

    def insert_data_model_to_table(self, data_class: Any, table_name: str, pk_key: str = "ID") -> int:
        """Inserts a data model instance into a table and returns the primary key."""
        if table_name is None:
            table_name = data_class.__class__.__name__.lower()

        # Get field values from data model
        field_dict = {field.name: getattr(data_class, field.name) for field in fields(data_class)}

        columns = ', '.join(field_dict.keys())
        values = ', '.join([f"'{v}'" if isinstance(v, str) else str(v) for v in field_dict.values()])

        query = f"INSERT INTO {self.schema}.{table_name} ({columns}) VALUES ({values}) RETURNING {pk_key};"

        try:
            df = self.run_sql_and_get_df(query)
            return df.collect()[0][0]
        except Exception as e:
            raise Exception(f"Failed to insert data model: {str(e)}")

    def write_dataframe_to_table(self, df: DataFrame, table_name: str, mode: str = "append"):
        """
        Writes a Spark DataFrame to a database table.

        :param df: Spark DataFrame to write
        :param table_name: Target table name
        :param mode: Write mode - 'append', 'overwrite', 'ignore', or 'error'
        """
        df.write \
            .format("jdbc") \
            .option("url", self.db_url) \
            .option("dbtable", f"{self.schema}.{table_name}") \
            .option("user", self.user) \
            .option("password", self.password) \
            .option("driver", "org.postgresql.Driver") \
            .mode(mode) \
            .save()
