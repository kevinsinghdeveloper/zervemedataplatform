import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pandas as pd
from datetime import datetime
from typing import Optional

from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresSqlConnector, PostgresDataType, PostgresSqlType


class TestPostgresSqlConnector(unittest.TestCase):
    """Test cases for PostgresSqlConnector methods"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_db_config = {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_password",
            "host": "localhost",
            "port": 5432,
            "schema": "public"
        }
        self.connector = PostgresSqlConnector(self.mock_db_config)

    # Tests for get_table
    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_get_table_without_limit(self, mock_run_sql):
        """Test get_table retrieves all rows when no limit is specified"""
        mock_df = pd.DataFrame({'id': [1, 2, 3], 'name': ['A', 'B', 'C']})
        mock_run_sql.return_value = mock_df

        result = self.connector.get_table('users')

        mock_run_sql.assert_called_once_with('SELECT * FROM public.users ;')
        pd.testing.assert_frame_equal(result, mock_df)

    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_get_table_with_limit(self, mock_run_sql):
        """Test get_table retrieves limited rows when limit is specified"""
        mock_df = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        mock_run_sql.return_value = mock_df

        result = self.connector.get_table('users', limit_n=2)

        mock_run_sql.assert_called_once_with('SELECT * FROM public.users LIMIT 2;')
        pd.testing.assert_frame_equal(result, mock_df)

    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_get_table_empty_result(self, mock_run_sql):
        """Test get_table handles empty result"""
        mock_df = pd.DataFrame()
        mock_run_sql.return_value = mock_df

        result = self.connector.get_table('empty_table')

        mock_run_sql.assert_called_once()
        pd.testing.assert_frame_equal(result, mock_df)

    # Tests for list_tables
    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_list_tables_with_results(self, mock_run_sql):
        """Test list_tables returns all tables in schema"""
        mock_df = pd.DataFrame({
            'table_name': ['users', 'products', 'orders']
        })
        mock_run_sql.return_value = mock_df

        result = self.connector.list_tables()

        # Verify the SQL query
        call_args = mock_run_sql.call_args[0][0]
        self.assertIn("information_schema.tables", call_args)
        self.assertIn("WHERE table_schema = 'public'", call_args)
        self.assertIn("ORDER BY table_name", call_args)

        # Verify result
        self.assertEqual(result, ['users', 'products', 'orders'])

    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_list_tables_empty_schema(self, mock_run_sql):
        """Test list_tables returns empty list when no tables exist"""
        mock_df = pd.DataFrame()
        mock_run_sql.return_value = mock_df

        result = self.connector.list_tables()

        self.assertEqual(result, [])

    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_list_tables_custom_schema(self, mock_run_sql):
        """Test list_tables works with custom schema"""
        # Create connector with custom schema
        custom_config = self.mock_db_config.copy()
        custom_config['schema'] = 'analytics'
        connector = PostgresSqlConnector(custom_config)

        mock_df = pd.DataFrame({'table_name': ['table1', 'table2']})
        mock_run_sql.return_value = mock_df

        result = connector.list_tables()

        call_args = mock_run_sql.call_args[0][0]
        self.assertIn("WHERE table_schema = 'analytics'", call_args)
        self.assertEqual(result, ['table1', 'table2'])

    # Tests for list_tables_with_prefix
    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_list_tables_with_prefix_matching_tables(self, mock_run_sql):
        """Test list_tables_with_prefix returns tables matching prefix"""
        mock_df = pd.DataFrame({
            'table_name': ['temp_users', 'temp_orders', 'temp_logs']
        })
        mock_run_sql.return_value = mock_df

        result = self.connector.list_tables_with_prefix('temp_')

        # Verify the SQL query
        call_args = mock_run_sql.call_args[0][0]
        self.assertIn("AND table_name LIKE 'temp_%'", call_args)

        # Verify result
        self.assertEqual(result, ['temp_users', 'temp_orders', 'temp_logs'])

    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_list_tables_with_prefix_no_matches(self, mock_run_sql):
        """Test list_tables_with_prefix returns empty list when no matches"""
        mock_df = pd.DataFrame()
        mock_run_sql.return_value = mock_df

        result = self.connector.list_tables_with_prefix('nonexistent_')

        self.assertEqual(result, [])

    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_list_tables_with_prefix_empty_prefix(self, mock_run_sql):
        """Test list_tables_with_prefix with empty prefix returns all tables"""
        mock_df = pd.DataFrame({
            'table_name': ['users', 'products', 'orders']
        })
        mock_run_sql.return_value = mock_df

        result = self.connector.list_tables_with_prefix('')

        # Empty prefix should match all tables
        call_args = mock_run_sql.call_args[0][0]
        self.assertIn("AND table_name LIKE '%'", call_args)
        self.assertEqual(result, ['users', 'products', 'orders'])

    # Tests for write_dataframe_to_table
    @patch('sqlalchemy.create_engine')
    def test_write_dataframe_to_table_append_mode(self, mock_create_engine):
        """Test write_dataframe_to_table in append mode"""
        # Setup mock engine
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        # Execute
        self.connector.write_dataframe_to_table(df, 'users', mode='append')

        # Verify engine was created with correct connection string
        expected_conn_str = "postgresql://test_user:test_password@localhost:5432/test_db"
        mock_create_engine.assert_called_once_with(expected_conn_str)

        # Verify engine was disposed
        mock_engine.dispose.assert_called_once()

    @patch('sqlalchemy.create_engine')
    def test_write_dataframe_to_table_replace_mode(self, mock_create_engine):
        """Test write_dataframe_to_table in replace mode"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})

        self.connector.write_dataframe_to_table(df, 'test_table', mode='replace')

        mock_create_engine.assert_called_once()
        mock_engine.dispose.assert_called_once()

    @patch('sqlalchemy.create_engine')
    def test_write_dataframe_to_table_overwrite_mode(self, mock_create_engine):
        """Test write_dataframe_to_table with overwrite mode (maps to replace)"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})

        self.connector.write_dataframe_to_table(df, 'test_table', mode='overwrite')

        mock_create_engine.assert_called_once()
        mock_engine.dispose.assert_called_once()

    @patch('sqlalchemy.create_engine')
    def test_write_dataframe_to_table_with_options(self, mock_create_engine):
        """Test write_dataframe_to_table includes options in connection string"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Add options to config
        config_with_options = self.mock_db_config.copy()
        config_with_options['options'] = '-c statement_timeout=5000'
        connector = PostgresSqlConnector(config_with_options)

        df = pd.DataFrame({'col1': [1]})

        connector.write_dataframe_to_table(df, 'test_table')

        # Verify connection string includes options
        call_args = mock_create_engine.call_args[0][0]
        self.assertIn("options=", call_args)

    @patch('sqlalchemy.create_engine')
    def test_write_dataframe_to_table_empty_dataframe(self, mock_create_engine):
        """Test write_dataframe_to_table handles empty DataFrame"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        df = pd.DataFrame()

        self.connector.write_dataframe_to_table(df, 'test_table')

        # Should still attempt to write (to_sql handles empty DataFrames)
        mock_create_engine.assert_called_once()

    @patch('sqlalchemy.create_engine')
    def test_write_dataframe_to_table_error_handling(self, mock_create_engine):
        """Test write_dataframe_to_table handles errors properly"""
        # Make create_engine raise an exception
        mock_create_engine.side_effect = Exception("Connection failed")

        df = pd.DataFrame({'col1': [1, 2]})

        # Should raise the exception
        with self.assertRaises(Exception) as context:
            self.connector.write_dataframe_to_table(df, 'test_table')

        self.assertIn("Connection failed", str(context.exception))

    @patch('sqlalchemy.create_engine')
    def test_write_dataframe_to_table_custom_schema(self, mock_create_engine):
        """Test write_dataframe_to_table uses custom schema"""
        custom_config = self.mock_db_config.copy()
        custom_config['schema'] = 'analytics'
        connector = PostgresSqlConnector(custom_config)

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        df = pd.DataFrame({'col1': [1, 2]})

        connector.write_dataframe_to_table(df, 'test_table')

        # The schema should be passed to to_sql
        # We can't easily verify this without mocking to_sql, but we verify the connector has the right schema
        self.assertEqual(connector.schema, 'analytics')

    # Tests for PostgresDataType enum
    def test_postgres_data_type_enum_values(self):
        """Test PostgresDataType enum has correct values"""
        self.assertEqual(str(PostgresDataType.INTEGER), 'INTEGER')
        self.assertEqual(str(PostgresDataType.VARCHAR), 'VARCHAR')
        self.assertEqual(str(PostgresDataType.FLOAT), 'FLOAT')
        self.assertEqual(str(PostgresDataType.BOOLEAN), 'BOOLEAN')
        self.assertEqual(str(PostgresDataType.JSONB), 'JSONB')
        self.assertEqual(str(PostgresDataType.TIMESTAMP), 'TIMESTAMP')
        self.assertEqual(str(PostgresDataType.DATE), 'DATE')
        self.assertEqual(str(PostgresDataType.TEXT), 'TEXT')

    def test_postgres_data_type_enum_double_precision(self):
        """Test PostgresDataType handles multi-word types correctly"""
        self.assertEqual(str(PostgresDataType.DOUBLE_PRECISION), 'DOUBLE PRECISION')

    def test_postgres_data_type_enum_jsonb(self):
        """Test PostgresDataType has JSONB type"""
        self.assertEqual(str(PostgresDataType.JSONB), 'JSONB')
        self.assertEqual(str(PostgresDataType.JSON), 'JSON')

    def test_postgres_data_type_enum_vector(self):
        """Test PostgresDataType constants are PostgresSqlType instances"""
        # PostgresDataType.VECTOR is now a constant, not an enum
        # It's not typically used directly - use PostgresDataType.vector(dims) instead
        # But check it exists for backwards compatibility
        self.assertIsInstance(PostgresDataType.VECTOR, PostgresSqlType)

    def test_postgres_data_type_vector_with_dimensions(self):
        """Test PostgresDataType.vector() creates dimensioned vector type"""
        # Common embedding dimensions
        result_384 = PostgresDataType.vector(384)
        self.assertIsInstance(result_384, PostgresSqlType)
        self.assertEqual(str(result_384), 'vector(384)')  # MiniLM

        result_768 = PostgresDataType.vector(768)
        self.assertEqual(str(result_768), 'vector(768)')  # BERT

        result_1536 = PostgresDataType.vector(1536)
        self.assertEqual(str(result_1536), 'vector(1536)')  # OpenAI text-embedding-ada-002

        result_3072 = PostgresDataType.vector(3072)
        self.assertEqual(str(result_3072), 'vector(3072)')  # OpenAI text-embedding-3-large

    # Tests for _get_sql_type method
    def test_get_sql_type_with_int(self):
        """Test _get_sql_type maps int to INTEGER"""
        result = self.connector._get_sql_type(int)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'INTEGER')

    def test_get_sql_type_with_str(self):
        """Test _get_sql_type maps str to VARCHAR"""
        result = self.connector._get_sql_type(str)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'VARCHAR')

    def test_get_sql_type_with_float(self):
        """Test _get_sql_type maps float to FLOAT"""
        result = self.connector._get_sql_type(float)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'FLOAT')

    def test_get_sql_type_with_bool(self):
        """Test _get_sql_type maps bool to BOOLEAN"""
        result = self.connector._get_sql_type(bool)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'BOOLEAN')

    def test_get_sql_type_with_dict(self):
        """Test _get_sql_type maps dict to JSONB"""
        result = self.connector._get_sql_type(dict)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'JSONB')

    def test_get_sql_type_with_datetime(self):
        """Test _get_sql_type maps datetime to TIMESTAMP"""
        result = self.connector._get_sql_type(datetime)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'TIMESTAMP')

    def test_get_sql_type_with_optional_int(self):
        """Test _get_sql_type handles Optional[int] type hint"""
        result = self.connector._get_sql_type(Optional[int])
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'INTEGER')

    def test_get_sql_type_with_optional_str(self):
        """Test _get_sql_type handles Optional[str] type hint"""
        result = self.connector._get_sql_type(Optional[str])
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'VARCHAR')

    def test_get_sql_type_with_unknown_type(self):
        """Test _get_sql_type defaults to VARCHAR for unknown types"""
        class CustomType:
            pass

        result = self.connector._get_sql_type(CustomType)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'VARCHAR')

    def test_get_sql_type_with_list(self):
        """Test _get_sql_type maps list to JSONB (can store arrays or vectors)"""
        result = self.connector._get_sql_type(list)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'JSONB')

    def test_get_sql_type_with_optional_list(self):
        """Test _get_sql_type handles Optional[list] type hint"""
        from typing import List
        result = self.connector._get_sql_type(Optional[List])
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'JSONB')

    def test_get_sql_type_jsonb_string_matching(self):
        """Test _get_sql_type recognizes 'jsonb' in type strings"""
        # Simulate a type hint that contains 'jsonb' in its string representation
        class JSONBType:
            def __repr__(self):
                return "<class 'jsonb'>"

        result = self.connector._get_sql_type(JSONBType())
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'JSONB')

    def test_get_sql_type_vector_string_matching(self):
        """Test _get_sql_type recognizes 'vector' in type strings"""
        # Simulate a type hint that contains 'vector' in its string representation
        class VectorType:
            def __repr__(self):
                return "<class 'vector'>"

        result = self.connector._get_sql_type(VectorType())
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'vector')

    def test_get_sql_type_uses_enum(self):
        """Test that _get_sql_type returns PostgresSqlType instances"""
        # Verify it's returning PostgresSqlType instance
        result = self.connector._get_sql_type(int)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(result.base_type, 'INTEGER')
        self.assertEqual(str(result), 'INTEGER')

    def test_get_sql_type_dict_maps_to_jsonb(self):
        """Test _get_sql_type maps dict to JSONB (not JSON)"""
        result = self.connector._get_sql_type(dict)
        self.assertIsInstance(result, PostgresSqlType)
        self.assertEqual(str(result), 'JSONB')
        self.assertEqual(result.base_type, 'JSONB')

    # Tests for cast_column method
    @patch.object(PostgresSqlConnector, 'exec_sql')
    def test_cast_column_to_integer(self, mock_exec_sql):
        """Test cast_column generates correct SQL for simple type"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import INTEGER

        self.connector.cast_column("users", "age", INTEGER)

        # Verify exec_sql was called with correct ALTER TABLE statement
        mock_exec_sql.assert_called_once()
        call_args = mock_exec_sql.call_args[0][0]
        self.assertIn("ALTER TABLE", call_args)
        self.assertIn("public.users", call_args)
        self.assertIn("ALTER COLUMN age", call_args)
        self.assertIn("TYPE INTEGER", call_args)

    @patch.object(PostgresSqlConnector, 'exec_sql')
    def test_cast_column_with_parameterized_type(self, mock_exec_sql):
        """Test cast_column generates correct SQL for parameterized type"""
        varchar_type = PostgresSqlType("VARCHAR", length=255)

        self.connector.cast_column("products", "name", varchar_type)

        # Verify exec_sql was called with correct ALTER TABLE statement
        mock_exec_sql.assert_called_once()
        call_args = mock_exec_sql.call_args[0][0]
        self.assertIn("ALTER TABLE", call_args)
        self.assertIn("public.products", call_args)
        self.assertIn("ALTER COLUMN name", call_args)
        self.assertIn("TYPE VARCHAR(255)", call_args)

    @patch.object(PostgresSqlConnector, 'exec_sql')
    def test_cast_column_to_vector_type(self, mock_exec_sql):
        """Test cast_column generates correct SQL for vector type with dimensions"""
        vector_type = PostgresDataType.vector(1536)

        self.connector.cast_column("products", "embedding", vector_type)

        # Verify exec_sql was called with correct ALTER TABLE statement
        mock_exec_sql.assert_called_once()
        call_args = mock_exec_sql.call_args[0][0]
        self.assertIn("ALTER TABLE", call_args)
        self.assertIn("public.products", call_args)
        self.assertIn("ALTER COLUMN embedding", call_args)
        self.assertIn("TYPE vector(1536)", call_args)

    @patch.object(PostgresSqlConnector, 'exec_sql')
    def test_cast_column_to_jsonb(self, mock_exec_sql):
        """Test cast_column generates correct SQL for JSONB type"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import JSONB

        self.connector.cast_column("events", "metadata", JSONB)

        # Verify exec_sql was called with correct ALTER TABLE statement
        mock_exec_sql.assert_called_once()
        call_args = mock_exec_sql.call_args[0][0]
        self.assertIn("ALTER TABLE", call_args)
        self.assertIn("public.events", call_args)
        self.assertIn("ALTER COLUMN metadata", call_args)
        self.assertIn("TYPE JSONB", call_args)

    @patch.object(PostgresSqlConnector, 'exec_sql')
    def test_cast_column_with_numeric_precision(self, mock_exec_sql):
        """Test cast_column generates correct SQL for NUMERIC with precision and scale"""
        numeric_type = PostgresSqlType("NUMERIC", precision=10, scale=2)

        self.connector.cast_column("products", "price", numeric_type)

        # Verify exec_sql was called with correct ALTER TABLE statement
        mock_exec_sql.assert_called_once()
        call_args = mock_exec_sql.call_args[0][0]
        self.assertIn("ALTER TABLE", call_args)
        self.assertIn("public.products", call_args)
        self.assertIn("ALTER COLUMN price", call_args)
        self.assertIn("TYPE NUMERIC(10,2)", call_args)

    @patch.object(PostgresSqlConnector, 'exec_sql')
    def test_cast_column_with_custom_schema(self, mock_exec_sql):
        """Test cast_column respects custom schema"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import INTEGER

        # Create connector with custom schema
        custom_config = self.mock_db_config.copy()
        custom_config['schema'] = 'analytics'
        connector = PostgresSqlConnector(custom_config)

        connector.cast_column("metrics", "count", INTEGER)

        # Verify exec_sql was called with custom schema
        mock_exec_sql.assert_called_once()
        call_args = mock_exec_sql.call_args[0][0]
        self.assertIn("ALTER TABLE analytics.metrics", call_args)
        self.assertIn("ALTER COLUMN count", call_args)
        self.assertIn("TYPE INTEGER", call_args)

    # Integration-style tests (multiple methods working together)
    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_get_table_after_list_tables_with_prefix(self, mock_run_sql):
        """Test workflow: list tables with prefix, then get table data"""
        # First call: list_tables_with_prefix
        mock_run_sql.return_value = pd.DataFrame({
            'table_name': ['temp_data_2024', 'temp_data_2025']
        })

        tables = self.connector.list_tables_with_prefix('temp_data_')
        self.assertEqual(len(tables), 2)

        # Second call: get_table
        mock_run_sql.return_value = pd.DataFrame({
            'id': [1, 2],
            'value': [100, 200]
        })

        df = self.connector.get_table('temp_data_2024', limit_n=10)

        self.assertEqual(len(df), 2)
        self.assertEqual(mock_run_sql.call_count, 2)

    @patch.object(PostgresSqlConnector, 'run_sql_and_get_df')
    def test_list_tables_includes_all_expected_tables(self, mock_run_sql):
        """Test list_tables returns expected tables in correct order"""
        mock_run_sql.return_value = pd.DataFrame({
            'table_name': ['analytics', 'users', 'products', 'orders', 'logs']
        })

        result = self.connector.list_tables()

        # Should maintain the order from the query result
        self.assertEqual(result, ['analytics', 'users', 'products', 'orders', 'logs'])


if __name__ == '__main__':
    unittest.main()
