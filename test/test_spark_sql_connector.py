import unittest
from unittest.mock import Mock, patch
from pyspark.sql import SparkSession, DataFrame
from dataclasses import dataclass

from zervedataplatform.connectors.sql_connectors.SparkSqlConnector import SparkSQLConnector


@dataclass
class TestDataModel:
    """Test data model for testing data class methods"""
    id: int
    name: str
    age: int
    email: str


class TestSparkSQLConnector(unittest.TestCase):
    """Test the SparkSQLConnector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_password",
            "schema": "public",
            "database_type": "postgres",  # Required for SqlConnectorHandler
            "spark_config": {
                "spark.jars": "/path/to/postgresql.jar",
                "spark.driver.memory": "2g"
            }
        }

    def tearDown(self):
        """Clean up after tests"""
        pass

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_initialization_with_db_config(self, mock_spark_session):
        """Test that SparkSQLConnector initializes correctly with db config"""
        # Mock SparkSession builder chain
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = Mock(spec=SparkSession)

        connector = SparkSQLConnector(self.db_config)

        # Verify database connection details are set
        self.assertEqual(connector.user, "test_user")
        self.assertEqual(connector.password, "test_password")
        self.assertEqual(connector.schema, "public")
        self.assertIn("jdbc:postgresql://", connector.db_url)
        self.assertIn("test_db", connector.db_url)

        # Verify Spark session was created with app name
        mock_spark_session.builder.appName.assert_called_once_with("SparkSQLSession")

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_get_spark(self, mock_spark_session):
        """Test get_spark returns the Spark session"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.get_spark()

        self.assertEqual(result, mock_spark)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_write_dataframe_to_table_append_mode(self, mock_spark_session):
        """Test writing DataFrame to table in append mode"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock DataFrame
        mock_df = Mock(spec=DataFrame)
        mock_writer = Mock()
        mock_df.write.format.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.save.return_value = None

        connector = SparkSQLConnector(self.db_config)
        connector.write_dataframe_to_table(mock_df, "test_table", mode="append")

        # Verify the write chain was called correctly
        mock_df.write.format.assert_called_once_with("jdbc")

        # Check that all required options were set
        option_calls = mock_writer.option.call_args_list
        option_dict = {call[0][0]: call[0][1] for call in option_calls}

        self.assertIn("url", option_dict)
        self.assertIn("jdbc:postgresql://", option_dict["url"])
        self.assertEqual(option_dict["dbtable"], "public.test_table")
        self.assertEqual(option_dict["user"], "test_user")
        self.assertEqual(option_dict["password"], "test_password")

        # Verify mode and save were called
        mock_writer.mode.assert_called_once_with("append")
        mock_writer.save.assert_called_once()

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_write_dataframe_to_table_overwrite_mode(self, mock_spark_session):
        """Test writing DataFrame to table in overwrite mode"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock DataFrame
        mock_df = Mock(spec=DataFrame)
        mock_writer = Mock()
        mock_df.write.format.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.save.return_value = None

        connector = SparkSQLConnector(self.db_config)
        connector.write_dataframe_to_table(mock_df, "test_table", mode="overwrite")

        # Verify mode was set to overwrite
        mock_writer.mode.assert_called_once_with("overwrite")
        mock_writer.save.assert_called_once()

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_run_sql_and_get_df(self, mock_spark_session):
        """Test running SQL query and getting DataFrame"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)

        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.run_sql_and_get_df("SELECT * FROM test_table")

        # Verify the read chain was called correctly
        mock_spark.read.format.assert_called_once_with("jdbc")

        # Check that query option was set
        option_calls = mock_reader.option.call_args_list
        option_dict = {call[0][0]: call[0][1] for call in option_calls}
        self.assertEqual(option_dict["query"], "SELECT * FROM test_table")

        mock_reader.load.assert_called_once()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_pull_data_from_table_with_filters(self, mock_spark_session):
        """Test pulling data from table with filters"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)

        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        columns = ["id", "name", "age"]
        filters = {"age": 25, "name": "John"}

        result = connector.pull_data_from_table("test_table", columns, filters)

        # Check that query was constructed with WHERE clause
        option_calls = mock_reader.option.call_args_list
        option_dict = {call[0][0]: call[0][1] for call in option_calls}
        query = option_dict["query"]

        self.assertIn("SELECT id, name, age FROM public.test_table", query)
        self.assertIn("WHERE", query)
        self.assertIn("age = '25'", query)
        self.assertIn("name = 'John'", query)

        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_check_if_table_exists(self, mock_spark_session):
        """Test checking if table exists"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_row = Mock()
        mock_row.__getitem__ = Mock(return_value=True)

        mock_df.collect.return_value = [mock_row]
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.check_if_table_exists("test_table")

        # Verify the query checks information_schema
        option_calls = mock_reader.option.call_args_list
        option_dict = {call[0][0]: call[0][1] for call in option_calls}
        query = option_dict["query"]

        self.assertIn("information_schema.tables", query)
        self.assertIn("test_table", query)
        self.assertTrue(result)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_get_table_row_count(self, mock_spark_session):
        """Test getting table row count"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_row = Mock()
        mock_row.__getitem__ = Mock(return_value=42)

        mock_df.collect.return_value = [mock_row]
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.get_table_row_count("test_table")

        # Verify the query uses COUNT(*)
        option_calls = mock_reader.option.call_args_list
        option_dict = {call[0][0]: call[0][1] for call in option_calls}
        query = option_dict["query"]

        self.assertIn("COUNT(*)", query)
        self.assertIn("test_table", query)
        self.assertEqual(result, 42)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_check_db_status_success(self, mock_spark_session):
        """Test database status check when connection is working"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_row = Mock()
        mock_row.__getitem__ = Mock(return_value=1)

        mock_df.collect.return_value = [mock_row]
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.check_db_status()

        self.assertTrue(result)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_check_db_status_failure(self, mock_spark_session):
        """Test database status check when connection fails"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()

        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.side_effect = Exception("Connection failed")

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.check_db_status()

        self.assertFalse(result)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_get_sql_type_basic_types(self, mock_spark_session):
        """Test _get_sql_type method with basic types"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)

        # Test basic type mappings (now delegated to helper connector)
        # These should match PostgresDataType enum values
        self.assertEqual(connector._get_sql_type(int), "INTEGER")
        self.assertEqual(connector._get_sql_type(float), "FLOAT")  # PostgresSqlConnector uses FLOAT
        self.assertEqual(connector._get_sql_type(str), "VARCHAR")  # PostgresSqlConnector uses VARCHAR
        self.assertEqual(connector._get_sql_type(bool), "BOOLEAN")

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SqlConnectorHandler')
    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_exec_sql_with_psycopg2(self, mock_spark_session, mock_handler):
        """Test exec_sql delegates to helper connector"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock helper connector
        mock_helper_connector = Mock()
        mock_handler.get_sql_connector.return_value = mock_helper_connector

        connector = SparkSQLConnector(self.db_config)
        test_query = "CREATE TABLE test (id INTEGER)"

        connector.exec_sql(test_query)

        # Verify helper connector's exec_sql was called with the query
        mock_helper_connector.exec_sql.assert_called_once_with(test_query)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SqlConnectorHandler')
    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_exec_sql_with_error_rollback(self, mock_spark_session, mock_handler):
        """Test exec_sql propagates errors from helper connector"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock helper connector to raise an error
        mock_helper_connector = Mock()
        mock_helper_connector.exec_sql.side_effect = Exception("SQL execution failed")
        mock_handler.get_sql_connector.return_value = mock_helper_connector

        connector = SparkSQLConnector(self.db_config)
        test_query = "INVALID SQL"

        # Should raise an exception
        with self.assertRaises(Exception) as context:
            connector.exec_sql(test_query)

        self.assertIn("SQL execution failed", str(context.exception))

        # Verify helper connector's exec_sql was called
        mock_helper_connector.exec_sql.assert_called_once_with(test_query)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SqlConnectorHandler')
    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_create_table_using_def(self, mock_spark_session, mock_handler):
        """Test creating table using column definitions"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock helper connector
        mock_helper_connector = Mock()
        mock_handler.get_sql_connector.return_value = mock_helper_connector

        connector = SparkSQLConnector(self.db_config)
        table_def = {
            "id": "INTEGER PRIMARY KEY",
            "name": "VARCHAR(255)",
            "age": "INTEGER"
        }

        # This will call exec_sql internally
        connector.create_table_using_def("test_table", table_def)

        # Verify helper connector's exec_sql was called
        mock_helper_connector.exec_sql.assert_called_once()

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SqlConnectorHandler')
    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_clone_table(self, mock_spark_session, mock_handler):
        """Test cloning a table"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock helper connector
        mock_helper_connector = Mock()
        mock_handler.get_sql_connector.return_value = mock_helper_connector

        connector = SparkSQLConnector(self.db_config)
        connector.clone_table("original_table", "cloned_table")

        # Verify helper connector's exec_sql was called
        mock_helper_connector.exec_sql.assert_called_once()

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SqlConnectorHandler')
    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_rename_table(self, mock_spark_session, mock_handler):
        """Test renaming a table"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock helper connector
        mock_helper_connector = Mock()
        mock_handler.get_sql_connector.return_value = mock_helper_connector

        connector = SparkSQLConnector(self.db_config)
        connector.rename_table("old_table", "new_table")

        # Verify helper connector's exec_sql was called
        mock_helper_connector.exec_sql.assert_called_once()

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_test_table_by_row_count_with_data(self, mock_spark_session):
        """Test table has rows"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_row = Mock()
        mock_row.__getitem__ = Mock(return_value=10)

        mock_df.collect.return_value = [mock_row]
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.test_table_by_row_count("test_table")

        self.assertTrue(result)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_test_table_by_row_count_empty(self, mock_spark_session):
        """Test table is empty"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_row = Mock()
        mock_row.__getitem__ = Mock(return_value=0)

        mock_df.collect.return_value = [mock_row]
        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.test_table_by_row_count("test_table")

        self.assertFalse(result)

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_get_distinct_values_from_single_col(self, mock_spark_session):
        """Test getting distinct values from a column"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_column = Mock()

        # Mock the column selection and RDD conversion
        mock_df.select.return_value = mock_column
        mock_column.rdd.flatMap.return_value.collect.return_value = ["value1", "value2", "value3"]

        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkSQLConnector(self.db_config)
        result = connector.get_distinct_values_from_single_col("status", "test_table")

        # Verify the query uses DISTINCT
        option_calls = mock_reader.option.call_args_list
        option_dict = {call[0][0]: call[0][1] for call in option_calls}
        query = option_dict["query"]

        self.assertIn("DISTINCT", query)
        self.assertIn("status", query)
        self.assertEqual(result, ["value1", "value2", "value3"])

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SqlConnectorHandler')
    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_create_table_ctas(self, mock_spark_session, mock_handler):
        """Test creating table using CTAS (CREATE TABLE AS SELECT)"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock helper connector
        mock_helper_connector = Mock()
        mock_handler.get_sql_connector.return_value = mock_helper_connector

        connector = SparkSQLConnector(self.db_config)
        inner_sql = "SELECT * FROM source_table WHERE age > 18"

        connector.create_table_ctas("new_table", inner_sql, include_print=False)

        # Verify helper connector's exec_sql was called
        mock_helper_connector.exec_sql.assert_called_once()

    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SqlConnectorHandler')
    @patch('zervedataplatform.connectors.sql_connectors.SparkSqlConnector.SparkSession')
    def test_append_to_table_insert_select(self, mock_spark_session, mock_handler):
        """Test appending data to table using INSERT INTO SELECT"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock helper connector
        mock_helper_connector = Mock()
        mock_handler.get_sql_connector.return_value = mock_helper_connector

        connector = SparkSQLConnector(self.db_config)
        inner_sql = "SELECT id, name FROM source_table"

        connector.append_to_table_insert_select("target_table", inner_sql, columnStr="id, name")

        # Verify helper connector's exec_sql was called
        mock_helper_connector.exec_sql.assert_called_once()


if __name__ == '__main__':
    unittest.main()
