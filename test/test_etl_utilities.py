import unittest
from unittest.mock import Mock, MagicMock, patch, call
from pyspark.sql import SparkSession, DataFrame
import pandas as pd

from zervedataplatform.utils.ETLUtilities import ETLUtilities
from zervedataplatform.model_transforms.db.PipelineRunConfig import PipelineRunConfig


class TestETLUtilities(unittest.TestCase):
    """Test cases for ETLUtilities"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock configuration
        self.mock_pipeline_config = Mock(spec=PipelineRunConfig)
        self.mock_pipeline_config.run_config = {
            'source_path': 's3://test-bucket/source',
            'xform_path': 's3://test-bucket/xform'
        }
        self.mock_pipeline_config.cloud_config = {
            'spark_config': {
                'spark.sql.shuffle.partitions': '2',
                'spark.default.parallelism': '2'
            }
        }
        self.mock_pipeline_config.db_config = {
            'database_type': 'postgres',
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db'
        }
        self.mock_pipeline_config.dest_db_config = {
            'database_type': 'postgres',
            'host': 'dest-host',
            'port': 5432,
            'database': 'dest_db'
        }

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_initialization(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test ETLUtilities initialization"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Verify SparkSession was created
        mock_spark_session.builder.appName.assert_called_once_with("SparkUtilitySession")

        # Verify cloud connector was initialized
        mock_cloud_connector.assert_called_once_with(self.mock_pipeline_config.cloud_config)

        # Verify SQL connectors were initialized for both source and dest
        self.assertEqual(mock_sql_connector.call_count, 2)
        mock_sql_connector.assert_any_call(self.mock_pipeline_config.db_config)
        mock_sql_connector.assert_any_call(self.mock_pipeline_config.dest_db_config)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_get_all_files_from_folder(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test get_all_files_from_folder returns files from cloud"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_cloud = Mock()
        mock_cloud.list_files.return_value = ['file1', 'file2', 'file3']
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.get_all_files_from_folder('s3://test-bucket/path')

        self.assertEqual(result, ['file1', 'file2', 'file3'])
        mock_cloud.list_files.assert_called_once_with('s3://test-bucket/path', 'folders')

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_get_all_files_from_folder_with_item_type(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test get_all_files_from_folder with custom item_type"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_cloud = Mock()
        mock_cloud.list_files.return_value = ['file1.parquet', 'file2.parquet']
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.get_all_files_from_folder('s3://test-bucket/path', item_type='files')

        mock_cloud.list_files.assert_called_once_with('s3://test-bucket/path', 'files')

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_check_all_files_consistency_in_folder_success(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test check_all_files_consistency_in_folder with valid files"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        # Mock DataFrame with valid data
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 10
        mock_df.isEmpty.return_value = False
        mock_df.columns = ['col1', 'col2', 'col3']

        mock_cloud = Mock()
        mock_cloud.list_files.return_value = ['file1', 'file2']
        mock_cloud.get_dataframe_from_cloud.return_value = mock_df
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        passed, errors = etl_util.check_all_files_consistency_in_folder('s3://test-bucket/folder')

        self.assertTrue(passed)
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors['file1'], [])
        self.assertEqual(errors['file2'], [])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_check_all_files_consistency_in_folder_empty_file(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test check_all_files_consistency_in_folder with empty file"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        # Mock empty DataFrame
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 0
        mock_df.isEmpty.return_value = True
        mock_df.columns = ['col1', 'col2']

        mock_cloud = Mock()
        mock_cloud.list_files.return_value = ['empty_file']
        mock_cloud.get_dataframe_from_cloud.return_value = mock_df
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        passed, errors = etl_util.check_all_files_consistency_in_folder('s3://test-bucket/folder')

        self.assertFalse(passed)
        self.assertIn("File content is empty", errors['empty_file'])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_check_all_files_consistency_in_folder_malformed(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test check_all_files_consistency_in_folder with malformed file"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        # Mock DataFrame with only one column (malformed)
        mock_df = Mock(spec=DataFrame)
        mock_df.count.return_value = 10
        mock_df.isEmpty.return_value = False
        mock_df.columns = ['col1']  # Only one column

        mock_cloud = Mock()
        mock_cloud.list_files.return_value = ['malformed_file']
        mock_cloud.get_dataframe_from_cloud.return_value = mock_df
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        passed, errors = etl_util.check_all_files_consistency_in_folder('s3://test-bucket/folder')

        self.assertFalse(passed)
        self.assertIn("File is malformed -- possibly delimiter issue?", errors['malformed_file'])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_get_latest_folder_using_config(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test get_latest_folder_using_config returns latest folder"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_cloud = Mock()
        folders = [
            's3://test-bucket/source/20231201_120000',
            's3://test-bucket/source/20231202_150000',
            's3://test-bucket/source/20231130_090000'
        ]
        mock_cloud.list_files.return_value = folders
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.get_latest_folder_using_config()

        self.assertEqual(result, 's3://test-bucket/source/20231202_150000')

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_move_folder_to_path(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test move_folder_to_path copies folder"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_cloud = Mock()
        mock_cloud.copy_folder.return_value = True
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.move_folder_to_path('source_path', 'dest_path')

        self.assertTrue(result)
        mock_cloud.copy_folder.assert_called_once_with('source_path', 'dest_path')

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_move_source_to_xform_location_using_config(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test move_source_to_xform_location_using_config"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_cloud = Mock()
        mock_cloud.copy_folder.return_value = True
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result, final_path = etl_util.move_source_to_xform_location_using_config(
            's3://test-bucket/source/20231201_120000'
        )

        self.assertTrue(result)
        self.assertEqual(final_path, 's3://test-bucket/xform/20231201_120000')

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_get_df_from_cloud(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test get_df_from_cloud retrieves DataFrame"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_df = Mock(spec=DataFrame)
        mock_cloud = Mock()
        mock_cloud.get_dataframe_from_cloud.return_value = mock_df
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.get_df_from_cloud('s3://test-bucket/file.parquet')

        self.assertEqual(result, mock_df)
        mock_cloud.get_dataframe_from_cloud.assert_called_once_with(file_path='s3://test-bucket/file.parquet')

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_write_df_to_table(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test write_df_to_table writes to database"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_sql = Mock()
        mock_sql_connector.return_value = mock_sql

        etl_util = ETLUtilities(self.mock_pipeline_config)

        mock_df = Mock(spec=DataFrame)
        etl_util.write_df_to_table(mock_df, 'test_table', mode='append')

        mock_sql.write_dataframe_to_table.assert_called_once_with(
            df=mock_df,
            table_name='test_table',
            mode='append'
        )

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_remove_tables_from_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test remove_tables_from_db drops multiple tables"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_sql = Mock()
        mock_sql_connector.return_value = mock_sql

        etl_util = ETLUtilities(self.mock_pipeline_config)

        tables = ['table1', 'table2', 'table3']
        etl_util.remove_tables_from_db(tables)

        self.assertEqual(mock_sql.drop_table.call_count, 3)
        mock_sql.drop_table.assert_has_calls([
            call('table1'),
            call('table2'),
            call('table3')
        ])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_convert_dict_to_spark_df(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test convert_dict_to_spark_df converts dict list to DataFrame"""
        # Create a real Spark session for this test
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = spark

        etl_util = ETLUtilities(self.mock_pipeline_config)

        data = [
            {'name': 'John', 'age': 30, 'tags': ['developer', 'python']},
            {'name': 'Jane', 'age': 25, 'tags': ['designer', 'ux']}
        ]

        result = etl_util.convert_dict_to_spark_df(data)

        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.count(), 2)

        # Check that array columns were converted to strings
        rows = result.collect()
        self.assertIn('developer,python', rows[0].tags)

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_add_column_to_spark_df(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test add_column_to_spark_df adds column to DataFrame"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = spark

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create test DataFrame
        data = [{'name': 'John'}, {'name': 'Jane'}]
        df = spark.createDataFrame(data)

        result = etl_util.add_column_to_spark_df(df, 'category', 'footwear')

        self.assertIn('category', result.columns)
        rows = result.collect()
        self.assertEqual(rows[0].category, 'footwear')
        self.assertEqual(rows[1].category, 'footwear')

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_upload_df(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test upload_df uploads DataFrame to cloud"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_cloud = Mock()
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        mock_df = Mock(spec=DataFrame)
        etl_util.upload_df(mock_df, 's3://test-bucket/output.parquet')

        mock_cloud.upload_data_frame_to_cloud.assert_called_once_with(
            df=mock_df,
            file_path='s3://test-bucket/output.parquet'
        )

    def test_get_latest_folder_from_list_static_method(self):
        """Test get_latest_folder_from_list returns latest folder"""
        folders = [
            's3://bucket/20231201_120000',
            's3://bucket/20231202_150000',
            's3://bucket/20231130_090000',
            's3://bucket/20231203_080000'
        ]

        result = ETLUtilities.get_latest_folder_from_list(folders)

        self.assertEqual(result, 's3://bucket/20231203_080000')

    def test_get_latest_folder_from_list_empty_list(self):
        """Test get_latest_folder_from_list with empty list returns None"""
        result = ETLUtilities.get_latest_folder_from_list([])

        self.assertIsNone(result)

    def test_get_latest_folder_from_list_single_folder(self):
        """Test get_latest_folder_from_list with single folder"""
        folders = ['s3://bucket/20231201_120000']

        result = ETLUtilities.get_latest_folder_from_list(folders)

        self.assertEqual(result, 's3://bucket/20231201_120000')

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_convert_pandas_to_spark_df(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test convert_pandas_to_spark_df converts Pandas DataFrame to Spark"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = spark

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create Pandas DataFrame
        pd_df = pd.DataFrame({
            'name': ['John', 'Jane', 'Bob'],
            'age': [30, 25, 35],
            'city': ['NYC', 'LA', 'Chicago']
        })

        result = etl_util.convert_pandas_to_spark_df(pd_df)

        self.assertIsInstance(result, DataFrame)
        self.assertEqual(result.count(), 3)
        self.assertEqual(len(result.columns), 3)

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_drop_db_tables_with_dest_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test drop_db_tables with use_dest_db=True drops from destination database"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        # Create two different mock SQL connectors
        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        tables = ['dest_table1', 'dest_table2']
        etl_util.drop_db_tables(tables, use_dest_db=True)

        # Verify dest_db_manager was used, not source
        self.assertEqual(mock_dest_sql.drop_table.call_count, 2)
        self.assertEqual(mock_source_sql.drop_table.call_count, 0)
        mock_dest_sql.drop_table.assert_has_calls([
            call('dest_table1'),
            call('dest_table2')
        ])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_drop_db_table_with_dest_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test drop_db_table with use_dest_db=True drops from destination database"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        etl_util.drop_db_table('dest_table', use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.drop_table.assert_called_once_with('dest_table')
        mock_source_sql.drop_table.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_find_all_tables_with_prefix_with_dest_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test find_all_tables_with_prefix with use_dest_db=True queries destination database"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_dest_sql.list_tables_with_prefix.return_value = ['dest_orders_2024', 'dest_orders_2025']
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.find_all_tables_with_prefix('dest_orders', use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.list_tables_with_prefix.assert_called_once_with('dest_orders')
        mock_source_sql.list_tables_with_prefix.assert_not_called()
        self.assertEqual(result, ['dest_orders_2024', 'dest_orders_2025'])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_write_df_to_table_with_dest_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test write_df_to_table with use_dest_db=True writes to destination database"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        mock_df = Mock(spec=DataFrame)
        etl_util.write_df_to_table(mock_df, 'dest_table', mode='append', use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.write_dataframe_to_table.assert_called_once_with(
            df=mock_df,
            table_name='dest_table',
            mode='append'
        )
        mock_source_sql.write_dataframe_to_table.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_remove_tables_from_db_with_dest_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test remove_tables_from_db with use_dest_db=True removes from destination database"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        tables = ['dest_temp1', 'dest_temp2', 'dest_temp3']
        etl_util.remove_tables_from_db(tables, use_dest_db=True)

        # Verify dest_db_manager was used
        self.assertEqual(mock_dest_sql.drop_table.call_count, 3)
        self.assertEqual(mock_source_sql.drop_table.call_count, 0)
        mock_dest_sql.drop_table.assert_has_calls([
            call('dest_temp1'),
            call('dest_temp2'),
            call('dest_temp3')
        ])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_data_movement_from_source_to_dest_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test complete data movement scenario: read from source, write to destination"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        # Mock cloud storage with DataFrame
        mock_cloud = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_cloud.get_dataframe_from_cloud.return_value = mock_df
        mock_cloud_connector.return_value = mock_cloud

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Simulate data movement:
        # 1. Read from cloud
        df = etl_util.get_df_from_cloud('s3://test-bucket/processed_data.parquet')

        # 2. Write to source DB for staging
        etl_util.write_df_to_table(df, 'staging_table', mode='overwrite', use_dest_db=False)

        # 3. Write to destination DB for final storage
        etl_util.write_df_to_table(df, 'production_table', mode='overwrite', use_dest_db=True)

        # Verify source DB was written to
        mock_source_sql.write_dataframe_to_table.assert_called_once_with(
            df=mock_df,
            table_name='staging_table',
            mode='overwrite'
        )

        # Verify destination DB was written to
        mock_dest_sql.write_dataframe_to_table.assert_called_once_with(
            df=mock_df,
            table_name='production_table',
            mode='overwrite'
        )

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_get_all_db_tables_from_source(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test get_all_db_tables returns tables from source database by default"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_source_sql.list_tables.return_value = ['source_table1', 'source_table2', 'source_table3']
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.get_all_db_tables()

        # Verify source_db_manager was used
        mock_source_sql.list_tables.assert_called_once()
        mock_dest_sql.list_tables.assert_not_called()
        self.assertEqual(result, ['source_table1', 'source_table2', 'source_table3'])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_get_all_db_tables_from_dest(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test get_all_db_tables with use_dest_db=True returns tables from destination database"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_dest_sql.list_tables.return_value = ['dest_table1', 'dest_table2', 'dest_analytics']
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.get_all_db_tables(use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.list_tables.assert_called_once()
        mock_source_sql.list_tables.assert_not_called()
        self.assertEqual(result, ['dest_table1', 'dest_table2', 'dest_analytics'])

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_read_db_table_to_df_from_source(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test read_db_table_to_df reads from source database by default"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_source_sql.get_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.read_db_table_to_df('test_table')

        # Verify source_db_manager was used
        mock_source_sql.get_table.assert_called_once_with('test_table', None)
        mock_dest_sql.get_table.assert_not_called()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_read_db_table_to_df_from_source_with_limit(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test read_db_table_to_df reads from source database with limit"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_source_sql.get_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.read_db_table_to_df('test_table', limit_n=100)

        # Verify source_db_manager was used with limit
        mock_source_sql.get_table.assert_called_once_with('test_table', 100)
        mock_dest_sql.get_table.assert_not_called()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_read_db_table_to_df_from_dest(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test read_db_table_to_df reads from destination database when use_dest_db=True"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_dest_sql.get_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.read_db_table_to_df('dest_table', use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.get_table.assert_called_once_with('dest_table', None)
        mock_source_sql.get_table.assert_not_called()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_read_db_table_to_df_from_dest_with_limit(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test read_db_table_to_df reads from destination database with limit"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_dest_sql.get_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        result = etl_util.read_db_table_to_df('dest_table', limit_n=50, use_dest_db=True)

        # Verify dest_db_manager was used with limit
        mock_dest_sql.get_table.assert_called_once_with('dest_table', 50)
        mock_source_sql.get_table.assert_not_called()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_pull_table_data_to_df_from_source(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test pull_table_data_to_df pulls from source database by default"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_source_sql.pull_data_from_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        columns = ['id', 'name', 'email']
        result = etl_util.pull_table_data_to_df('users', columns)

        # Verify source_db_manager was used
        mock_source_sql.pull_data_from_table.assert_called_once_with('users', columns, None)
        mock_dest_sql.pull_data_from_table.assert_not_called()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_pull_table_data_to_df_from_source_with_filters(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test pull_table_data_to_df pulls from source database with filters"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_source_sql.pull_data_from_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        columns = ['id', 'name', 'email']
        filters = {'status': 'active', 'role': 'admin'}
        result = etl_util.pull_table_data_to_df('users', columns, filters)

        # Verify source_db_manager was used with filters
        mock_source_sql.pull_data_from_table.assert_called_once_with('users', columns, filters)
        mock_dest_sql.pull_data_from_table.assert_not_called()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_pull_table_data_to_df_from_dest(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test pull_table_data_to_df pulls from destination database when use_dest_db=True"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_dest_sql.pull_data_from_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        columns = ['product_id', 'product_name', 'price']
        result = etl_util.pull_table_data_to_df('products', columns, use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.pull_data_from_table.assert_called_once_with('products', columns, None)
        mock_source_sql.pull_data_from_table.assert_not_called()
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_pull_table_data_to_df_from_dest_with_filters(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test pull_table_data_to_df pulls from destination database with filters"""
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_df = Mock(spec=DataFrame)
        mock_dest_sql.pull_data_from_table.return_value = mock_df
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        columns = ['order_id', 'customer_id', 'total']
        filters = {'status': 'completed', 'payment_method': 'credit_card'}
        result = etl_util.pull_table_data_to_df('orders', columns, filters, use_dest_db=True)

        # Verify dest_db_manager was used with filters
        mock_dest_sql.pull_data_from_table.assert_called_once_with('orders', columns, filters)
        mock_source_sql.pull_data_from_table.assert_not_called()
        self.assertEqual(result, mock_df)

    def test_combine_columns_to_text_basic(self):
        """Test combine_columns_to_text with basic settings"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        # Create test DataFrame
        data = [
            {"title": "Product A", "description": "Great product", "brand": "Nike"},
            {"title": "Product B", "description": "Amazing item", "brand": "Adidas"},
        ]
        df = spark.createDataFrame(data)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["title", "description", "brand"],
            output_column="combined_text"
        )

        # Check that combined_text column exists
        self.assertIn("combined_text", result.columns)

        # Check values
        rows = result.collect()
        self.assertIn("Product A", rows[0].combined_text)
        self.assertIn("Great product", rows[0].combined_text)
        self.assertIn("Nike", rows[0].combined_text)

        spark.stop()

    def test_combine_columns_to_text_with_labels(self):
        """Test combine_columns_to_text with add_labels=True"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        data = [
            {"product_title": "Shoes", "brand": "Nike", "color": "Red"}
        ]
        df = spark.createDataFrame(data)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["product_title", "brand", "color"],
            add_labels=True
        )

        rows = result.collect()
        combined_text = rows[0].combined_text

        # Check that labels are added
        self.assertIn("Product Title:", combined_text)
        self.assertIn("Brand:", combined_text)
        self.assertIn("Color:", combined_text)
        self.assertIn("Shoes", combined_text)
        self.assertIn("Nike", combined_text)
        self.assertIn("Red", combined_text)

        spark.stop()

    def test_combine_columns_to_text_with_nulls_filtered(self):
        """Test combine_columns_to_text filters out null values"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        data = [
            {"title": "Product A", "description": None, "brand": "Nike"},
            {"title": "Product B", "description": "Good", "brand": None},
        ]
        df = spark.createDataFrame(data)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["title", "description", "brand"],
            filter_empty=True
        )

        rows = result.collect()

        # First row should have title and brand, but not description
        self.assertIn("Product A", rows[0].combined_text)
        self.assertIn("Nike", rows[0].combined_text)
        # Should not have empty string markers
        self.assertNotIn("..", rows[0].combined_text.replace(". . ", ".."))

        # Second row should have title and description, but not brand
        self.assertIn("Product B", rows[1].combined_text)
        self.assertIn("Good", rows[1].combined_text)

        spark.stop()

    def test_combine_columns_to_text_without_filtering(self):
        """Test combine_columns_to_text with filter_empty=False"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        from pyspark.sql.types import StructType, StructField, StringType
        schema = StructType([
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("brand", StringType(), True)
        ])

        data = [
            {"title": "Product A", "description": None, "brand": "Nike"}
        ]
        df = spark.createDataFrame(data, schema=schema)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["title", "description", "brand"],
            filter_empty=False
        )

        rows = result.collect()
        # With filter_empty=False, empty values are included
        # This should have multiple separators next to each other
        self.assertIn("Product A", rows[0].combined_text)
        self.assertIn("Nike", rows[0].combined_text)

        spark.stop()

    def test_combine_columns_to_text_custom_separator(self):
        """Test combine_columns_to_text with custom separator"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        data = [
            {"col1": "A", "col2": "B", "col3": "C"}
        ]
        df = spark.createDataFrame(data)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["col1", "col2", "col3"],
            separator=" | "
        )

        rows = result.collect()
        self.assertEqual(rows[0].combined_text, "A | B | C")

        spark.stop()

    def test_combine_columns_to_text_custom_output_column(self):
        """Test combine_columns_to_text with custom output column name"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        data = [
            {"col1": "A", "col2": "B"}
        ]
        df = spark.createDataFrame(data)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["col1", "col2"],
            output_column="custom_column"
        )

        # Check that custom column exists
        self.assertIn("custom_column", result.columns)
        self.assertNotIn("combined_text", result.columns)

        rows = result.collect()
        self.assertIn("A", rows[0].custom_column)
        self.assertIn("B", rows[0].custom_column)

        spark.stop()

    def test_combine_columns_to_text_preserves_other_columns(self):
        """Test that combine_columns_to_text preserves other columns"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        data = [
            {"id": 1, "title": "Product A", "description": "Great", "price": 100.0}
        ]
        df = spark.createDataFrame(data)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["title", "description"]
        )

        # All original columns should still exist
        self.assertIn("id", result.columns)
        self.assertIn("title", result.columns)
        self.assertIn("description", result.columns)
        self.assertIn("price", result.columns)
        self.assertIn("combined_text", result.columns)

        rows = result.collect()
        self.assertEqual(rows[0].id, 1)
        self.assertEqual(rows[0].price, 100.0)

        spark.stop()

    def test_combine_columns_to_text_with_all_empty_values(self):
        """Test combine_columns_to_text when all values are empty"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        from pyspark.sql.types import StructType, StructField, StringType
        schema = StructType([
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("brand", StringType(), True)
        ])

        data = [
            {"title": None, "description": None, "brand": None}
        ]
        df = spark.createDataFrame(data, schema=schema)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["title", "description", "brand"],
            filter_empty=True
        )

        rows = result.collect()
        # When all values are empty, result should be empty string
        self.assertEqual(rows[0].combined_text, "")

        spark.stop()

    def test_combine_columns_to_text_multiple_rows(self):
        """Test combine_columns_to_text with multiple rows"""
        spark = SparkSession.builder.appName("TestApp").master("local[1]").getOrCreate()

        data = [
            {"title": "Product 1", "brand": "Nike"},
            {"title": "Product 2", "brand": "Adidas"},
            {"title": "Product 3", "brand": "Puma"}
        ]
        df = spark.createDataFrame(data)

        result = ETLUtilities.combine_columns_to_text(
            df,
            columns=["title", "brand"],
            separator=" - "
        )

        rows = result.collect()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].combined_text, "Product 1 - Nike")
        self.assertEqual(rows[1].combined_text, "Product 2 - Adidas")
        self.assertEqual(rows[2].combined_text, "Product 3 - Puma")

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_cast_db_col_on_source_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test cast_db_col casts column type on source database by default"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create a vector type for casting
        vector_type = PostgresDataType.vector(1536)
        etl_util.cast_db_col('products', 'embedding', vector_type)

        # Verify source_db_manager was used
        mock_source_sql.cast_column.assert_called_once_with('products', 'embedding', vector_type)
        mock_dest_sql.cast_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_cast_db_col_on_dest_db(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test cast_db_col casts column type on destination database when use_dest_db=True"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create a JSONB type for casting
        jsonb_type = PostgresDataType.JSONB
        etl_util.cast_db_col('analytics_data', 'metadata', jsonb_type, use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.cast_column.assert_called_once_with('analytics_data', 'metadata', jsonb_type)
        mock_source_sql.cast_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_create_index_col_on_source_db_ivfflat_cosine(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test create_index_col creates IVFFlat cosine index on source database by default"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create an IVFFlat cosine index
        index = PostgresDataType.ivfflat_cosine(table='products', column='embedding', lists=100)
        etl_util.create_index_col(index)

        # Verify source_db_manager was used
        mock_source_sql.create_index_column.assert_called_once_with(index)
        mock_dest_sql.create_index_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_create_index_col_on_dest_db_ivfflat_cosine(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test create_index_col creates IVFFlat cosine index on destination database when use_dest_db=True"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create an IVFFlat cosine index for destination DB
        index = PostgresDataType.ivfflat_cosine(table='products', column='embedding', lists=150)
        etl_util.create_index_col(index, use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.create_index_column.assert_called_once_with(index)
        mock_source_sql.create_index_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_create_index_col_on_source_db_ivfflat_l2(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test create_index_col creates IVFFlat L2 index on source database"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create an IVFFlat L2 index
        index = PostgresDataType.ivfflat_l2(table='images', column='features_vector', lists=200)
        etl_util.create_index_col(index)

        # Verify source_db_manager was used
        mock_source_sql.create_index_column.assert_called_once_with(index)
        mock_dest_sql.create_index_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_create_index_col_on_dest_db_hnsw_cosine(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test create_index_col creates HNSW cosine index on destination database"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create an HNSW cosine index for destination DB
        index = PostgresDataType.hnsw_cosine(table='documents', column='text_embedding', m=16, ef_construction=64)
        etl_util.create_index_col(index, use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.create_index_column.assert_called_once_with(index)
        mock_source_sql.create_index_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_create_index_col_on_source_db_hnsw_l2(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test create_index_col creates HNSW L2 index on source database"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create an HNSW L2 index
        index = PostgresDataType.hnsw_l2(table='videos', column='feature_embedding', m=24, ef_construction=128)
        etl_util.create_index_col(index)

        # Verify source_db_manager was used
        mock_source_sql.create_index_column.assert_called_once_with(index)
        mock_dest_sql.create_index_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_create_index_col_on_dest_db_ivfflat_ip(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test create_index_col creates IVFFlat inner product index on destination database"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create an IVFFlat inner product index for destination DB
        index = PostgresDataType.ivfflat_ip(table='recommendations', column='user_embedding', lists=75)
        etl_util.create_index_col(index, use_dest_db=True)

        # Verify dest_db_manager was used
        mock_dest_sql.create_index_column.assert_called_once_with(index)
        mock_source_sql.create_index_column.assert_not_called()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_create_index_col_on_source_db_hnsw_ip(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test create_index_col creates HNSW inner product index on source database"""
        from zervedataplatform.connectors.sql_connectors.PostgresSqlConnector import PostgresDataType

        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        mock_source_sql = Mock()
        mock_dest_sql = Mock()
        mock_sql_connector.side_effect = [mock_source_sql, mock_dest_sql]

        etl_util = ETLUtilities(self.mock_pipeline_config)

        # Create an HNSW inner product index
        index = PostgresDataType.hnsw_ip(table='search_results', column='query_embedding', m=32, ef_construction=256)
        etl_util.create_index_col(index)

        # Verify source_db_manager was used
        mock_source_sql.create_index_column.assert_called_once_with(index)
        mock_dest_sql.create_index_column.assert_not_called()

    # Tests for union_spark_df method
    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_basic(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df with basic DataFrames"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        spark = SparkSession.builder.appName("test").getOrCreate()

        # Create two DataFrames with same schema
        df1 = spark.createDataFrame([
            {"product_id": 1, "price": 10.0, "name": "Product A"},
            {"product_id": 2, "price": 20.0, "name": "Product B"}
        ])

        df2 = spark.createDataFrame([
            {"product_id": 3, "price": 30.0, "name": "Product C"},
            {"product_id": 4, "price": 40.0, "name": "Product D"}
        ])

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "product_id": "int",
            "price": "float",
            "name": "string"
        }

        result = etl_util.union_spark_df([df1, df2], schema_map)

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.count(), 4)
        self.assertEqual(len(result.columns), 3)

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_with_missing_columns(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df handles DataFrames with missing columns"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        spark = SparkSession.builder.appName("test").getOrCreate()

        # df1 has all columns
        df1 = spark.createDataFrame([
            {"product_id": 1, "price": 10.0, "name": "Product A"}
        ])

        # df2 is missing 'name' column
        df2 = spark.createDataFrame([
            {"product_id": 2, "price": 20.0}
        ])

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "product_id": "int",
            "price": "float",
            "name": "string"
        }

        result = etl_util.union_spark_df([df1, df2], schema_map)

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.count(), 2)
        self.assertEqual(len(result.columns), 3)

        # Check that missing column was filled with null
        rows = result.collect()
        self.assertIsNone(rows[1]["name"])

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_with_type_casting(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df casts columns to correct types"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        spark = SparkSession.builder.appName("test").getOrCreate()

        # df1 has price as int
        df1 = spark.createDataFrame([
            {"product_id": 1, "price": 10, "name": "Product A"}
        ])

        # df2 has price as float
        df2 = spark.createDataFrame([
            {"product_id": 2, "price": 20.5, "name": "Product B"}
        ])

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "product_id": "int",
            "price": "float",
            "name": "string"
        }

        result = etl_util.union_spark_df([df1, df2], schema_map)

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.count(), 2)

        # Check data types
        price_type = [field.dataType.simpleString() for field in result.schema.fields if field.name == "price"][0]
        self.assertIn("float", price_type.lower())

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_with_extra_columns(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df removes extra columns not in schema_map"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        spark = SparkSession.builder.appName("test").getOrCreate()

        # df1 has an extra column 'description'
        df1 = spark.createDataFrame([
            {"product_id": 1, "price": 10.0, "name": "Product A", "description": "Extra"}
        ])

        df2 = spark.createDataFrame([
            {"product_id": 2, "price": 20.0, "name": "Product B"}
        ])

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "product_id": "int",
            "price": "float",
            "name": "string"
        }

        result = etl_util.union_spark_df([df1, df2], schema_map)

        # Verify result only has columns from schema_map
        self.assertIsNotNone(result)
        self.assertEqual(len(result.columns), 3)
        self.assertNotIn("description", result.columns)

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_multiple_dataframes(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df with more than 2 DataFrames"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        spark = SparkSession.builder.appName("test").getOrCreate()

        df1 = spark.createDataFrame([{"id": 1, "value": "A"}])
        df2 = spark.createDataFrame([{"id": 2, "value": "B"}])
        df3 = spark.createDataFrame([{"id": 3, "value": "C"}])
        df4 = spark.createDataFrame([{"id": 4, "value": "D"}])

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "id": "int",
            "value": "string"
        }

        result = etl_util.union_spark_df([df1, df2, df3, df4], schema_map)

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.count(), 4)

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_empty_list(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df returns None for empty DataFrame list"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "product_id": "int",
            "price": "float"
        }

        result = etl_util.union_spark_df([], schema_map)

        # Verify result is None
        self.assertIsNone(result)

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_single_dataframe(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df with a single DataFrame"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        spark = SparkSession.builder.appName("test").getOrCreate()

        df = spark.createDataFrame([
            {"product_id": 1, "price": 10.0, "name": "Product A"}
        ])

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "product_id": "int",
            "price": "float",
            "name": "string"
        }

        result = etl_util.union_spark_df([df], schema_map)

        # Verify result
        self.assertIsNotNone(result)
        self.assertEqual(result.count(), 1)
        self.assertEqual(len(result.columns), 3)

        spark.stop()

    @patch('zervedataplatform.utils.ETLUtilities.SparkSQLConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkCloudConnector')
    @patch('zervedataplatform.utils.ETLUtilities.SparkSession')
    def test_union_spark_df_preserves_column_order(self, mock_spark_session, mock_cloud_connector, mock_sql_connector):
        """Test union_spark_df preserves column order from schema_map"""
        # Set up SparkSession mock
        mock_spark = MagicMock()
        mock_spark_session.builder.appName.return_value.config.return_value.config.return_value.getOrCreate.return_value = mock_spark

        spark = SparkSession.builder.appName("test").getOrCreate()

        # df with columns in different order
        df1 = spark.createDataFrame([
            {"name": "Product A", "product_id": 1, "price": 10.0}
        ])

        df2 = spark.createDataFrame([
            {"price": 20.0, "name": "Product B", "product_id": 2}
        ])

        etl_util = ETLUtilities(self.mock_pipeline_config)

        schema_map = {
            "product_id": "int",
            "price": "float",
            "name": "string"
        }

        result = etl_util.union_spark_df([df1, df2], schema_map)

        # Verify column order matches schema_map
        self.assertEqual(result.columns, list(schema_map.keys()))

        spark.stop()


if __name__ == '__main__':
    unittest.main()