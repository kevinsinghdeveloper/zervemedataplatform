import unittest
from unittest.mock import Mock, MagicMock, patch
from pyspark.sql import SparkSession, DataFrame

from zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector import SparkCloudConnector


class TestSparkCloudConnector(unittest.TestCase):
    """Test the SparkCloudConnector class"""

    def setUp(self):
        """Set up test fixtures"""
        self.cloud_config = {
            "bucket": "test-bucket",
            "prefix": "test-prefix",
            "spark_config": {
                "spark.hadoop.fs.s3a.access.key": "test-access-key",
                "spark.hadoop.fs.s3a.secret.key": "test-secret-key",
                "spark.hadoop.fs.s3a.endpoint": "s3.amazonaws.com"
            }
        }

    def tearDown(self):
        """Clean up after tests"""
        pass

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_initialization_with_spark_config(self, mock_spark_session):
        """Test that SparkCloudConnector initializes correctly with spark config"""
        # Mock SparkSession builder chain
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = Mock(spec=SparkSession)

        connector = SparkCloudConnector(self.cloud_config)

        # Verify bucket and prefix are set
        self.assertEqual(connector.bucket, "test-bucket")
        self.assertEqual(connector.prefix, "test-prefix")

        # Verify Spark session was created with app name
        mock_spark_session.builder.appName.assert_called_once_with("SparkCloudSession")

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_initialization_with_defaults(self, mock_spark_session):
        """Test initialization with default values when config is minimal"""
        # Mock SparkSession builder chain
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = Mock(spec=SparkSession)

        minimal_config = {}
        connector = SparkCloudConnector(minimal_config)

        # Verify defaults are used
        self.assertEqual(connector.bucket, "default-bucket")
        self.assertEqual(connector.prefix, "")

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_get_spark(self, mock_spark_session):
        """Test get_spark returns the Spark session"""
        # Mock SparkSession
        mock_spark = Mock(spec=SparkSession)
        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.get_spark()

        self.assertEqual(result, mock_spark)

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_get_dataframe_from_cloud_csv(self, mock_spark_session):
        """Test reading CSV from cloud storage"""
        # Mock Spark session and DataFrame reader
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

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.get_dataframe_from_cloud("path/to/file.csv", file_format="csv", sep="|")

        # Verify the read chain was called correctly
        mock_spark.read.format.assert_called_once_with("csv")
        self.assertEqual(mock_reader.option.call_count, 2)  # header and sep
        mock_reader.load.assert_called_once_with("s3a://test-bucket/path/to/file.csv")
        self.assertEqual(result, mock_df)

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_get_dataframe_from_cloud_with_full_s3_path(self, mock_spark_session):
        """Test reading from cloud with full s3a:// path"""
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

        connector = SparkCloudConnector(self.cloud_config)
        full_path = "s3a://other-bucket/path/to/file.csv"
        result = connector.get_dataframe_from_cloud(full_path, file_format="csv")

        # Verify it uses the full path as-is
        mock_reader.load.assert_called_once_with(full_path)

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_upload_data_frame_to_cloud_success(self, mock_spark_session):
        """Test uploading DataFrame to cloud storage successfully"""
        # Mock Spark session and JVM
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        # Setup JVM mock chain
        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x)))

        # Mock filesystem operations
        mock_fs.exists.return_value = False
        mock_fs.rename.return_value = True

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

        connector = SparkCloudConnector(self.cloud_config)
        connector.upload_data_frame_to_cloud(mock_df, "output/path", file_format="csv", sep="|")

        # Verify write was called
        mock_df.write.format.assert_called_once_with("csv")
        mock_writer.save.assert_called_once()

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_upload_data_frame_to_cloud_write_error(self, mock_spark_session, mock_utility):
        """Test handling of write error during DataFrame upload"""
        mock_spark = Mock(spec=SparkSession)

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        # Mock DataFrame that raises error on write
        mock_df = Mock(spec=DataFrame)
        mock_writer = Mock()
        mock_df.write.format.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.save.side_effect = Exception("Write failed")

        connector = SparkCloudConnector(self.cloud_config)
        connector.upload_data_frame_to_cloud(mock_df, "output/path")

        # Verify error was logged
        mock_utility.error_log.assert_called_once()
        self.assertIn("Error writing DataFrame", str(mock_utility.error_log.call_args))

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_get_dict_from_cloud(self, mock_spark_session):
        """Test reading JSON file as dictionary from cloud"""
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)

        # Mock DataFrame conversion to pandas and dict
        mock_pandas_df = Mock()
        mock_pandas_df.to_dict.return_value = {"key": "value"}
        mock_df.toPandas.return_value = mock_pandas_df
        mock_df.rdd.isEmpty.return_value = False

        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.get_dict_from_cloud("path/to/data.json")

        # Verify the result
        self.assertEqual(result, {"key": "value"})
        mock_pandas_df.to_dict.assert_called_once_with(orient="records")

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_get_dict_from_cloud_empty(self, mock_spark_session):
        """Test reading empty JSON file returns empty dict"""
        mock_spark = Mock(spec=SparkSession)
        mock_reader = Mock()
        mock_df = Mock(spec=DataFrame)

        # Mock empty DataFrame
        mock_df.rdd.isEmpty.return_value = True

        mock_spark.read.format.return_value = mock_reader
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = mock_df

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.get_dict_from_cloud("path/to/empty.json")

        self.assertEqual(result, {})

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_upload_dict_to_cloud(self, mock_spark_session):
        """Test uploading dictionary as JSON to cloud"""
        mock_spark = MagicMock()
        mock_df = Mock(spec=DataFrame)
        mock_writer = Mock()

        mock_spark.createDataFrame.return_value = mock_df
        mock_df.write.format.return_value = mock_writer
        mock_writer.option.return_value = mock_writer
        mock_writer.mode.return_value = mock_writer
        mock_writer.save.return_value = None

        # Mock JVM for upload
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()
        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x)))
        mock_fs.exists.return_value = False
        mock_fs.rename.return_value = True

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        test_dict = {"key": "value", "number": 42}
        connector.upload_dict_to_cloud(test_dict, "output/data.json")

        # Verify DataFrame was created from dict
        mock_spark.createDataFrame.assert_called_once_with([test_dict])
        mock_df.write.format.assert_called_once_with("json")

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_list_files_with_relative_prefix(self, mock_spark_session, mock_utility):
        """Test listing files with relative prefix"""
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        # Setup JVM mock
        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x)))

        # Mock file status objects
        mock_file1 = Mock()
        mock_file1.getPath.return_value.toString.return_value = "s3a://test-bucket/folder/file1.csv"
        mock_file2 = Mock()
        mock_file2.getPath.return_value.toString.return_value = "s3a://test-bucket/folder/file2.csv"

        mock_fs.exists.return_value = True
        mock_fs.listStatus.return_value = [mock_file1, mock_file2]

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.list_files("folder/", item_type="all")

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertIn("s3a://test-bucket/folder/file1.csv", result)
        self.assertIn("s3a://test-bucket/folder/file2.csv", result)

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_list_files_with_full_s3_path(self, mock_spark_session, mock_utility):
        """Test listing files with full S3 path"""
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x)))

        mock_fs.exists.return_value = True
        mock_fs.listStatus.return_value = []

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        full_path = "s3a://other-bucket/data/"
        result = connector.list_files(full_path)

        # Verify it uses the full path
        self.assertEqual(result, [])

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_list_files_path_does_not_exist(self, mock_spark_session, mock_utility):
        """Test listing files when path doesn't exist"""
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x)))

        mock_fs.exists.return_value = False

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.list_files("nonexistent/path/")

        # Verify warning was logged and empty list returned
        mock_utility.warning_log.assert_called_once()
        self.assertEqual(result, [])

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_list_files_filter_by_type(self, mock_spark_session, mock_utility):
        """Test filtering files by type (files vs folders)"""
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs

        # Create a Path mock that preserves the path string
        def create_path_mock(path_str):
            mock_path = Mock()
            mock_path.toUri.return_value = path_str
            mock_path.__str__ = Mock(return_value=path_str)
            return mock_path

        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=create_path_mock)

        # Mock file status objects - one file, one folder
        mock_file = Mock()
        mock_file.getPath.return_value.toString.return_value = "s3a://test-bucket/file.csv"
        mock_folder = Mock()
        mock_folder.getPath.return_value.toString.return_value = "s3a://test-bucket/folder/"

        mock_file_status = Mock()
        mock_file_status.isFile.return_value = True
        mock_file_status.isDirectory.return_value = False

        mock_folder_status = Mock()
        mock_folder_status.isFile.return_value = False
        mock_folder_status.isDirectory.return_value = True

        mock_fs.exists.return_value = True
        mock_fs.listStatus.return_value = [mock_file, mock_folder]

        # Use the actual path string from the Path mock's toUri to determine type
        def get_file_status(path):
            path_str = str(path.toUri.return_value) if hasattr(path, 'toUri') else str(path)
            return mock_file_status if "file.csv" in path_str else mock_folder_status

        mock_fs.getFileStatus.side_effect = get_file_status

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)

        # Test files only
        result_files = connector.list_files("", item_type="files")
        self.assertEqual(len(result_files), 1)
        self.assertIn("file.csv", result_files[0])

        # Test folders only
        result_folders = connector.list_files("", item_type="folders")
        self.assertEqual(len(result_folders), 1)
        self.assertIn("folder", result_folders[0])

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_copy_folder_source_not_exists(self, mock_spark_session, mock_utility):
        """Test copy folder when source doesn't exist"""
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x)))

        mock_fs.exists.return_value = False

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.copy_folder("nonexistent/path", "dest/path")

        self.assertFalse(result)
        mock_utility.warning_log.assert_called_once()
        self.assertIn("does not exist", str(mock_utility.warning_log.call_args))

    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_copy_folder_source_not_directory(self, mock_spark_session, mock_utility):
        """Test copy folder when source is not a directory"""
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x)))

        mock_source_status = Mock()
        mock_source_status.isDirectory.return_value = False

        mock_fs.exists.return_value = True
        mock_fs.getFileStatus.return_value = mock_source_status

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.copy_folder("file.csv", "dest/path")

        self.assertFalse(result)
        mock_utility.warning_log.assert_called_once()
        self.assertIn("not a directory", str(mock_utility.warning_log.call_args))


    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.Utility')
    @patch('zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector.SparkSession')
    def test_move_folder_destination_exists(self, mock_spark_session, mock_utility):
        """Test move folder when destination already exists"""
        mock_spark = MagicMock()
        mock_jvm = Mock()
        mock_fs = Mock()
        mock_hadoop_conf = Mock()

        mock_spark._jsc.hadoopConfiguration.return_value = mock_hadoop_conf
        mock_spark._jvm = mock_jvm
        mock_jvm.org.apache.hadoop.fs.FileSystem.get.return_value = mock_fs
        mock_jvm.org.apache.hadoop.fs.Path = Mock(side_effect=lambda x: Mock(toUri=Mock(return_value=x), getParent=Mock(return_value=Mock())))

        mock_source_status = Mock()
        mock_source_status.isDirectory.return_value = True

        mock_fs.exists.return_value = True  # Both source and dest exist
        mock_fs.getFileStatus.return_value = mock_source_status

        mock_builder = Mock()
        mock_spark_session.builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = mock_spark

        connector = SparkCloudConnector(self.cloud_config)
        result = connector.move_folder("source/path", "dest/path")

        self.assertFalse(result)
        mock_utility.warning_log.assert_called()
        self.assertIn("already exists", str(mock_utility.warning_log.call_args))


if __name__ == '__main__':
    unittest.main()