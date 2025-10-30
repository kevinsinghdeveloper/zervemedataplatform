from datetime import datetime
from typing import Union, Any, List

from pyspark import Row
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, concat_ws, col, coalesce, concat, array, array_remove

from zervedataplatform.connectors.cloud_storage_connectors.SparkCloudConnector import SparkCloudConnector
from zervedataplatform.connectors.sql_connectors.SparkSqlConnector import SparkSQLConnector
from zervedataplatform.model_transforms.db.PipelineRunConfig import PipelineRunConfig
from zervedataplatform.utils.Utility import Utility


class ETLUtilities:
    def __init__(self, pipeline_run_config: PipelineRunConfig):
        self.__run_config = pipeline_run_config.run_config

        spark_builder = SparkSession.builder.appName("SparkUtilitySession")

        for key, value in pipeline_run_config.cloud_config['spark_config'].items():
            spark_builder = spark_builder.config(key, value)

        self.__spark_utilities = spark_builder.getOrCreate()

        self.__spark_cloud_manager = None
        self.__spark_source_db_manager = None
        self.__spark_dest_db_manager = None

        if pipeline_run_config.cloud_config:
            self.__spark_cloud_manager = SparkCloudConnector(pipeline_run_config.cloud_config)

        if pipeline_run_config.db_config:
            self.__spark_source_db_manager = SparkSQLConnector(pipeline_run_config.db_config)

        if pipeline_run_config.dest_db_config:
            self.__spark_dest_db_manager = SparkSQLConnector(pipeline_run_config.dest_db_config)


    ''' PRE VALIDATION FUNCTIONS '''

    def get_all_files_from_folder(self,  prefix: str, item_type='folders'):
        files = self.__spark_cloud_manager.list_files(prefix, item_type)

        return files

    def drop_db_tables(self, table_names: list[str], use_dest_db: bool = False):
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        for table in table_names:
            Utility.log(f"Dropping table... {table}")
            db_manager.drop_table(table)

    def drop_db_table(self, table_name: str, use_dest_db: bool = False):
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        db_manager.drop_table(table_name)

    def find_all_tables_with_prefix(self, prefix: str, use_dest_db: bool = False) -> list[str]:
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        tables = db_manager.list_tables_with_prefix(prefix)
        return tables

    def check_all_files_consistency_in_folder(self, folder: str) -> tuple[bool, dict[str, list[Any]]]:
        # for each df
        errors = {}

        sub_folders = self.get_all_files_from_folder(folder)
        for sf in sub_folders:
            Utility.log(f"Checking file consistency for... {sf}")

            errors[sf] = []
            df = self.__spark_cloud_manager.get_dataframe_from_cloud(file_path=sf)

            if df.count() == 0 or df.isEmpty():
                errors[sf].append("File content is empty")

            cols = len(df.columns)

            if not cols > 1:
                errors[sf].append("File is malformed -- possibly delimiter issue?")

        passed = any(False if len(errors[f]) > 0 else True for f in errors)

        return passed, errors

    def get_latest_folder_using_config(self) -> Union[str | None]:
        source_location = self.__run_config['source_path']
        items = self.get_all_files_from_folder(source_location)

        return ETLUtilities.get_latest_folder_from_list(items)

    # Extraction
    def move_folder_to_path(self, source, destination):
        # result = self.__spark_cloud_manager.move_folder(source, destination)
        result = self.__spark_cloud_manager.copy_folder(source, destination)
        return result

    def move_source_to_xform_location_using_config(self, source: str):
        xform_path = self.__run_config['xform_path']

        folder = source.split("/")[-1]  # get folder name

        final_path = xform_path + "/" + folder
        result = self.move_folder_to_path(source, final_path)

        return result, final_path

    def get_df_from_cloud(self, path):
        return self.__spark_cloud_manager.get_dataframe_from_cloud(file_path=path)

    def write_df_to_table(self, df: DataFrame, table_name: str, mode: str = "overwrite", use_dest_db: bool = False):
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        db_manager.write_dataframe_to_table(df=df, table_name=table_name, mode=mode)

    def read_db_table_to_df(self, table_name, limit_n: int = None, use_dest_db: bool = False):
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        return db_manager.get_table(table_name, limit_n)

    def pull_table_data_to_df(self, table_name, columns: [str], filters: dict | None = None, use_dest_db: bool = False):
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        return db_manager.pull_data_from_table(table_name, columns, filters)

    def remove_tables_from_db(self, table_names: list[str], use_dest_db: bool = False):
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        for table in table_names:
            db_manager.drop_table(table)

    def convert_dict_to_spark_df(self, data: list[dict]):
        # return self.__spark_utilities.createDataFrame([Row(**item) for item in data])
        # Create initial DataFrame
        df = self.__spark_utilities.createDataFrame([Row(**item) for item in data])

        # Convert array columns to comma-separated strings
        for column_name in df.columns:
            if df.schema[column_name].dataType.typeName() == "array":
                df = df.withColumn(column_name, concat_ws(",", col(column_name)))

        return df

    def add_column_to_spark_df(self, df, column_name: str, column_value: any):
        return df.withColumn(column_name, lit(column_value))

    def upload_df(self, df: DataFrame, file_name: str):
        self.__spark_cloud_manager.upload_data_frame_to_cloud(df=df, file_path=file_name)

    def get_all_db_tables(self, use_dest_db: bool = False):
        db_manager = self.__spark_dest_db_manager if use_dest_db else self.__spark_source_db_manager
        return db_manager.list_tables()

    @staticmethod
    def get_latest_folder_from_list(folders: [str]) -> Union[str | None]:
        def extract_timestamp(folder: str) -> datetime:
            date_str = folder.split('/')[-1]  # Gets the timestamp part
            return datetime.strptime(date_str, '%Y%m%d_%H%M%S')

        if not folders:
            return None

        return max(folders, key=extract_timestamp)

    def convert_pandas_to_spark_df(self, pd_xform_df):
        return self.__spark_utilities.createDataFrame(pd_xform_df)

    @staticmethod
    def get_current_datestamp() -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    @staticmethod
    def combine_columns_to_text(
            df: DataFrame,
            columns: List[str],
            output_column: str = "combined_text",
            separator: str = ". ",
            add_labels: bool = False,
            filter_empty: bool = True
    ) -> DataFrame:
        """
        Combine multiple columns into a single text column for embeddings.

        Parameters:
        -----------
        df : DataFrame
            Input PySpark DataFrame
        columns : List[str]
            List of column names to combine
        output_column : str, default="combined_text"
            Name of the output column
        separator : str, default=". "
            Separator between column values
        add_labels : bool, default=False
            Whether to add column name as label (e.g., "Brand: Nike")
        filter_empty : bool, default=True
            Whether to filter out empty/null values

        Returns:
        --------
        DataFrame
            DataFrame with new combined text column

        Example:
        --------
        df_combined = combine_columns_to_text(
            df,
            ["product_title", "description", "category", "brand"],
            add_labels=True
        )
        """

        if filter_empty:
            # Build array and remove empty strings
            if add_labels:
                text_parts = [
                    concat(
                        lit(f"{col_name.replace('_', ' ').title()}: "),
                        coalesce(col(col_name), lit(""))
                    )
                    for col_name in columns
                ]
            else:
                text_parts = [coalesce(col(col_name), lit("")) for col_name in columns]

            return df.withColumn(
                "_temp_array",
                array(*text_parts)
            ).withColumn(
                output_column,
                concat_ws(separator, array_remove(col("_temp_array"), ""))
            ).drop("_temp_array")

        else:
            # Simple concatenation without filtering
            if add_labels:
                text_parts = [
                    concat(
                        lit(f"{col_name.replace('_', ' ').title()}: "),
                        coalesce(col(col_name), lit(""))
                    )
                    for col_name in columns
                ]
            else:
                text_parts = [coalesce(col(col_name), lit("")) for col_name in columns]

            return df.withColumn(
                output_column,
                concat_ws(separator, *text_parts)
            )


