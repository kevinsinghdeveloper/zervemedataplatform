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
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db'
        }
        self.mock_pipeline_config.dest_db_config = {
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


if __name__ == '__main__':
    unittest.main()