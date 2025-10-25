from typing import Dict, List
from pyspark.sql import SparkSession, DataFrame
from zervedataplatform.abstractions.connectors.CloudConnector import CloudConnector
from zervedataplatform.utils.Utility import Utility


class SparkCloudConnector(CloudConnector):
    def __init__(self, cloud_config: dict):
        """
        Initialize SparkCloudConnector with a dedicated Spark session for cloud storage operations.

        :param cloud_config: Dictionary containing cloud storage settings (bucket, S3 settings, etc.).
        """
        super().__init__(cloud_config)

        # Create Spark session specifically for cloud operations
        self._spark = SparkSession.builder.appName("SparkCloudSession").getOrCreate()

        # Cloud storage details
        self.bucket = cloud_config.get("bucket", "default-bucket")
        self.prefix = cloud_config.get("prefix", "")

    def get_spark(self):
        """Returns the Spark session for cloud operations."""
        return self._spark

    def get_dataframe_from_cloud(self, file_path: str, file_format: str = "csv", sep: str="|") -> DataFrame:
        """
        Reads a file from cloud storage (S3, GCS, etc.) and returns a Spark DataFrame.

        :param sep:
        :param file_path: The file path in cloud storage.
        :param file_format: File format (parquet, csv, json).
        :return: Spark DataFrame.
        """
        full_path = f"s3a://{self.bucket}/{file_path}" if not file_path.startswith("s3a://") else file_path
        return self.get_spark().read.format(file_format).option("header", "true").option("sep", sep).load(full_path)

    # def upload_data_frame_to_cloud(self, df: DataFrame, file_path: str, file_format: str = "csv", sep: str="|"):
    #     """
    #     Uploads a Spark DataFrame to cloud storage.
    #     DO not include extension in file name
    #     :param sep:
    #     :param df: The Spark DataFrame to be uploaded.
    #     :param file_path: The destination path in the cloud storage bucket.
    #     :param file_format: File format (parquet, csv, json).
    #     """
    #     full_path = f"s3a://{self.bucket}/{file_path}" if not file_path.startswith("s3a://") else file_path
    #     df.write.format(file_format).option("sep", sep).option("header", "true").mode("overwrite").save(full_path)
    def upload_data_frame_to_cloud(self, df: DataFrame, file_path: str, file_format: str = "csv", sep: str = "|"):
        """
        Uploads a Spark DataFrame to cloud storage safely, avoiding conflicts when overwriting.

        :param df: The Spark DataFrame to be uploaded.
        :param file_path: The destination path in cloud storage.
        :param file_format: File format (parquet, csv, json).
        :param sep: Separator for CSV files (default: "|").
        """
        full_path = f"s3a://{self.bucket}/{file_path}" if not file_path.startswith("s3a://") else file_path
        temp_path = full_path + "_temp"  # Temporary location

        try:
            df.write.format(file_format) \
                .option("sep", sep) \
                .option("header", "true") \
                .mode("overwrite") \
                .save(temp_path)
        except Exception as e:
            Utility.error_log(f"Error writing DataFrame to temporary path {temp_path}: {str(e)}")
            return

        hadoop_conf = self.get_spark()._jsc.hadoopConfiguration()
        fs = self.get_spark()._jvm.org.apache.hadoop.fs.FileSystem.get(
            self.get_spark()._jvm.org.apache.hadoop.fs.Path(full_path).toUri(), hadoop_conf
        )

        final_path = self.get_spark()._jvm.org.apache.hadoop.fs.Path(full_path)
        temp_s3_path = self.get_spark()._jvm.org.apache.hadoop.fs.Path(temp_path)

        if fs.exists(final_path):
            fs.delete(final_path, True)

        fs.rename(temp_s3_path, final_path)

    def get_dict_from_cloud(self, file_path: str) -> Dict:
        """
        Reads a JSON file from cloud storage and returns a dictionary.

        :param file_path: Path to the JSON file.
        :return: Dictionary representation of JSON data.
        """
        df = self.get_dataframe_from_cloud(file_path, "json")
        return df.toPandas().to_dict(orient="records") if not df.rdd.isEmpty() else {}

    def upload_dict_to_cloud(self, data: Dict, file_path: str):
        """
        Uploads a dictionary as a JSON file to cloud storage.

        :param data: The dictionary to upload.
        :param file_path: The destination path in cloud storage.
        """
        df = self.get_spark().createDataFrame([data])
        self.upload_data_frame_to_cloud(df, file_path, "json")

    def list_files(self, prefix: str = "", item_type: str = "all") -> List[str]:
        """
        Lists files and folders in a cloud storage bucket.

        :param prefix: The folder path to list contents from. Can be:
                       - A full S3 path (e.g., "s3a://my-bucket/folder/")
                       - A relative prefix (e.g., "folder/")
        :param item_type: "files", "folders", or "all" (default).
        :return: List of file/folder paths.
        """
        # Handle cases where prefix is a full S3 path or a relative path
        if prefix.startswith("s3a://"):
            full_prefix = prefix  # Already a full S3 path
        else:
            full_prefix = f"s3a://{self.bucket}/{prefix}" if prefix else f"s3a://{self.bucket}/"

        hadoop_conf = self.get_spark()._jsc.hadoopConfiguration()
        fs = self.get_spark()._jvm.org.apache.hadoop.fs.FileSystem.get(
            self.get_spark()._jvm.org.apache.hadoop.fs.Path(full_prefix).toUri(), hadoop_conf
        )
        path = self.get_spark()._jvm.org.apache.hadoop.fs.Path(full_prefix)

        if not fs.exists(path):
            Utility.warning_log(f"Path {full_prefix} does not exist.")
            return []

        status = fs.listStatus(path)
        items = [f.getPath().toString() for f in status]

        if item_type == "files":
            return [item for item in items if
                    fs.getFileStatus(self.get_spark()._jvm.org.apache.hadoop.fs.Path(item)).isFile()]
        elif item_type == "folders":
            return [item for item in items if
                    fs.getFileStatus(self.get_spark()._jvm.org.apache.hadoop.fs.Path(item)).isDirectory()]

        return items  # Return both files and folders

    def copy_folder(self, source_path: str, destination_path: str) -> bool:
        """
        Copies a folder from source to destination location in cloud storage.
        Preserves the source folder.

        :param source_path: Source folder path (can be full S3 path or relative path)
        :param destination_path: Destination folder path (can be full S3 path or relative path)
        :return: True if successful, False otherwise
        """
        # Handle cases where paths are full S3 paths or relative paths
        source_full_path = source_path if source_path.startswith("s3a://") else f"s3a://{self.bucket}/{source_path}"
        dest_full_path = destination_path if destination_path.startswith(
            "s3a://") else f"s3a://{self.bucket}/{destination_path}"

        try:
            hadoop_conf = self.get_spark()._jsc.hadoopConfiguration()
            fs = self.get_spark()._jvm.org.apache.hadoop.fs.FileSystem.get(
                self.get_spark()._jvm.org.apache.hadoop.fs.Path(source_full_path).toUri(),
                hadoop_conf
            )

            source = self.get_spark()._jvm.org.apache.hadoop.fs.Path(source_full_path)
            destination = self.get_spark()._jvm.org.apache.hadoop.fs.Path(dest_full_path)

            # Check if source exists and is a directory
            if not fs.exists(source):
                Utility.warning_log(f"Source path {source_full_path} does not exist.")
                return False

            if not fs.getFileStatus(source).isDirectory():
                Utility.warning_log(f"Source path {source_full_path} is not a directory.")
                return False

            # Check if destination parent directory exists
            dest_parent = destination.getParent()
            if not fs.exists(dest_parent):
                fs.mkdirs(dest_parent)

            # Check if destination already exists
            if fs.exists(destination):
                Utility.warning_log(f"Destination path {dest_full_path} already exists.")
                return False

            # Use FileUtil to copy directories
            file_util = self.get_spark()._jvm.org.apache.hadoop.fs.FileUtil
            success = file_util.copy(fs, source, fs, destination, False, True, hadoop_conf)

            if success:
                Utility.log(f"Successfully copied folder from {source_full_path} to {dest_full_path}")
            else:
                Utility.warning_log(f"Failed to copy folder from {source_full_path} to {dest_full_path}")

            return success

        except Exception as e:
            print(f"Error copying folder: {str(e)}")
            return False

    def move_folder(self, source_path: str, destination_path: str) -> bool:
        """
        Moves a folder from source to destination location in cloud storage.

        :param source_path: Source folder path (can be full S3 path or relative path)
        :param destination_path: Destination folder path (can be full S3 path or relative path)
        :return: True if successful, False otherwise
        """
        # Handle cases where paths are full S3 paths or relative paths
        source_full_path = source_path if source_path.startswith("s3a://") else f"s3a://{self.bucket}/{source_path}"
        dest_full_path = destination_path if destination_path.startswith(
            "s3a://") else f"s3a://{self.bucket}/{destination_path}"

        try:
            hadoop_conf = self.get_spark()._jsc.hadoopConfiguration()
            fs = self.get_spark()._jvm.org.apache.hadoop.fs.FileSystem.get(
                self.get_spark()._jvm.org.apache.hadoop.fs.Path(source_full_path).toUri(),
                hadoop_conf
            )

            source = self.get_spark()._jvm.org.apache.hadoop.fs.Path(source_full_path)
            destination = self.get_spark()._jvm.org.apache.hadoop.fs.Path(dest_full_path)

            # Check if source exists and is a directory
            if not fs.exists(source):
                Utility.warning_log(f"Source path {source_full_path} does not exist.")
                return False

            if not fs.getFileStatus(source).isDirectory():
                Utility.warning_log(f"Source path {source_full_path} is not a directory.")
                return False

            # Check if destination parent directory exists
            dest_parent = destination.getParent()
            if not fs.exists(dest_parent):
                fs.mkdirs(dest_parent)

            # Check if destination already exists
            if fs.exists(destination):
                Utility.warning_log(f"Destination path {dest_full_path} already exists.")
                return False

            # Perform the move operation
            success = fs.rename(source, destination)

            if success:
                Utility.log(f"Successfully moved folder from {source_full_path} to {dest_full_path}")
            else:
                Utility.warning_log(f"Failed to move folder from {source_full_path} to {dest_full_path}")

            return success

        except Exception as e:
            print(f"Error moving folder: {str(e)}")
            return False
