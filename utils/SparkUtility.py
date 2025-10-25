from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit
from pyspark.sql.utils import AnalysisException


class SparkUtility:
    def __init__(self, app_name: str, s3_bucket: str, aws_credentials: dict, pg_credentials: dict = None):
        """
        Initialize Spark Utility class with S3 and PostgreSQL configurations.

        :param app_name: Name of the Spark application
        :param s3_bucket: Default S3 bucket for file operations
        :param pg_credentials: Dictionary containing PostgreSQL credentials
        :param aws_credentials: Dictionary containing AWS credentials (access_key, secret_key)
        """
        self.s3_bucket = s3_bucket

        # Initialize Spark Session with required configurations
        # self.spark = SparkSession.builder \
        #     .appName(app_name) \
        #     .config("spark.hadoop.fs.s3a.access.key", aws_credentials["access_key"]) \
        #     .config("spark.hadoop.fs.s3a.secret.key", aws_credentials["secret_key"]) \
        #     .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
        #     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        #     .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")\
        #     .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        #     .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
        #     .getOrCreate()

        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.hadoop.fs.s3a.access.key", aws_credentials["access_key"]) \
            .config("spark.hadoop.fs.s3a.secret.key", aws_credentials["secret_key"]) \
            .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.hadoop.fs.defaultFS", f"s3a://{s3_bucket}/")\
            .config("spark.hadoop.fs.s3a.connection.maximum", "100") \
            .config("spark.hadoop.fs.s3a.fast.upload", "true") \
            .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider") \
            .config("spark.jars.packages",
                    "org.apache.hadoop:hadoop-aws:3.3.4,"
                    "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
                    "org.postgresql:postgresql:42.6.0") \
            .getOrCreate()

        if pg_credentials:
            # PostgreSQL connection details
            self.pg_host = pg_credentials["host"]
            self.pg_db = pg_credentials["database"]
            self.pg_user = pg_credentials["user"]
            self.pg_password = pg_credentials["password"]
            self.pg_port = pg_credentials.get("port", 5432)
            self.pg_url = f"jdbc:postgresql://{self.pg_host}:{self.pg_port}/{self.pg_db}"

    def get_spark(self):
        return self.spark

    ## ---- S3 OPERATIONS USING SPARK ---- ##
    def list_files_in_path(self, s3_path: str = "", include_folders: bool = True):
        """
        List files and optionally folders in a given S3 path using Spark.

        :param s3_path: S3 path (e.g., "s3a://my-bucket/folder/"). If empty, lists root of the bucket.
        :param include_folders: If True, lists both files and folders. If False, only lists files.
        :return: List of paths in S3.
        """
        if not s3_path:
            s3_path = f"s3a://{self.s3_bucket}/"  # Default to bucket root
        elif not s3_path.startswith("s3a://"):
            s3_path = f"s3a://{self.s3_bucket}/{s3_path}"  # Ensure correct format

        hadoop_conf = self.spark._jsc.hadoopConfiguration()
        fs = self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            self.spark._jvm.org.apache.hadoop.fs.Path(s3_path).toUri(), hadoop_conf
        )

        path = self.spark._jvm.org.apache.hadoop.fs.Path(s3_path)

        if fs.exists(path):
            status = fs.listStatus(path)
            if include_folders:
                return [f.getPath().toString() for f in status]  # Files + Folders
            else:
                return [f.getPath().toString() for f in status if f.isFile()]  # Only Files
        else:
            print(f"Path {s3_path} does not exist.")
            return []

    def move_s3_file(self, source_path: str, dest_path: str):
        """
        Move a file from one S3 location to another using Spark.

        :param source_path: Full S3 path of the source file (s3://bucket/key)
        :param dest_path: Full S3 path of the destination file (s3://bucket/key)
        """
        df = self.spark.read.format("csv").load(source_path)  # Load as DataFrame
        df.write.format("csv").mode("overwrite").save(dest_path)  # Save to new location
        self.spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            self.spark._jsc.hadoopConfiguration()
        ).delete(self.spark._jvm.org.apache.hadoop.fs.Path(source_path), True)
        print(f"Moved {source_path} to {dest_path}")

    ## ---- POSTGRES OPERATIONS USING SPARK ---- ##

    def fetch_dataframe(self, query: str) -> DataFrame:
        """
        Run a query and return a Spark DataFrame.

        :param query: SQL query string
        :return: Spark DataFrame
        """
        return self.spark.read \
            .format("jdbc") \
            .option("url", self.pg_url) \
            .option("query", query) \
            .option("user", self.pg_user) \
            .option("password", self.pg_password) \
            .option("driver", "org.postgresql.Driver") \
            .load()

    def write_dataframe(self, df: DataFrame, table_name: str, mode: str = "append"):
        """
        Write a Spark DataFrame to PostgreSQL, ensuring schema compatibility.

        :param df: Spark DataFrame to write
        :param table_name: PostgreSQL table name
        :param mode: Write mode ("append", "overwrite")
        """
        try:
            # Try to load the existing table schema from PostgreSQL
            try:
                existing_df = self.spark.read \
                    .format("jdbc") \
                    .option("url", self.pg_url) \
                    .option("dbtable", table_name) \
                    .option("user", self.pg_user) \
                    .option("password", self.pg_password) \
                    .option("driver", "org.postgresql.Driver") \
                    .load()

                existing_columns = set(existing_df.columns)
                df_columns = set(df.columns)

                # Drop extra columns in df that don't exist in table
                extra_columns = df_columns - existing_columns
                if extra_columns:
                    print(f"Dropping extra columns: {extra_columns}")
                    df = df.drop(*extra_columns)

                # Add missing columns with NULL values
                missing_columns = existing_columns - df_columns
                for col_name in missing_columns:
                    df = df.withColumn(col_name, lit(None))  # Add missing columns with NULL

            except AnalysisException:
                print(f"Table {table_name} does not exist. It will be created.")

            # Write the DataFrame
            df.write \
                .format("jdbc") \
                .option("url", self.pg_url) \
                .option("dbtable", table_name) \
                .option("user", self.pg_user) \
                .option("password", self.pg_password) \
                .option("driver", "org.postgresql.Driver") \
                .mode(mode) \
                .save()
            print(f"Data written successfully to {table_name} with mode {mode}")

        except Exception as e:
            print(f"Error writing DataFrame: {e}")
