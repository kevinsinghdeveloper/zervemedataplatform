from dataclasses import fields

import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession
from typing import Union, List, Dict, Optional

from pyspark.sql.functions import col
from pyspark.sql.types import StructField, StringType, StructType

from data_platform.abstractions.types.models.LLMData import LLMData


class DataTransformationUtility:
    def __init__(self):
        pass

    @staticmethod
    def get_spark_session(app_name: str = 'SparkApp') -> SparkSession:
        return SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()

    @staticmethod
    def read_csv(file_path: str, spark: bool, delimiter: str = '|', **kwargs) -> Union[pd.DataFrame, SparkDataFrame]:
        if spark:
            spark_session = DataTransformationUtility.get_spark_session()
            return spark_session.read.csv(file_path, header=True, inferSchema=True, sep=delimiter, **kwargs)
        else:
            return pd.read_csv(file_path, sep=delimiter, **kwargs)

    @staticmethod
    def dedupe_df(df: Union[pd.DataFrame, SparkDataFrame], subset: List[str] = None) -> Union[
        pd.DataFrame, SparkDataFrame]:

        if isinstance(df, pd.DataFrame):
            return df.drop_duplicates(subset=subset)
        elif isinstance(df, SparkDataFrame):
            return df.dropDuplicates(subset=subset)
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def filter_df(df: Union[pd.DataFrame, SparkDataFrame], mask) -> Union[pd.DataFrame, SparkDataFrame]:
        if isinstance(df, pd.DataFrame):
            return df[mask]
        elif isinstance(df, SparkDataFrame):
            if not isinstance(mask, list):
                raise TypeError("For Spark DataFrame, mask must be a list of conditions.")
            # Combine multiple conditions using 'reduce' and 'and'
            from functools import reduce
            combined_condition = reduce(lambda a, b: a | b, mask)
            return df.filter(combined_condition)
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def filter_df_where_cols_are_not_null(df: Union[pd.DataFrame, SparkDataFrame], cols: List[str]) -> Union[
        pd.DataFrame, SparkDataFrame]:
        if isinstance(df, pd.DataFrame):
            # Create a mask for pandas DataFrame
            mask = pd.Series([False] * len(df))
            for col_name in cols:
                mask |= df[col_name].notnull()
            return df[mask]
        elif isinstance(df, SparkDataFrame):
            # Create conditions for Spark DataFrame
            conditions = [col(col_name).isNotNull() for col_name in cols]
            # Combine multiple conditions using 'reduce' and 'or'
            from functools import reduce
            combined_condition = reduce(lambda a, b: a | b, conditions)
            return df.filter(combined_condition)
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def drop_columns(df: Union[pd.DataFrame, SparkDataFrame], cols: List[str]) -> Union[pd.DataFrame, SparkDataFrame]:
        if isinstance(df, pd.DataFrame):
            existing_cols = [col for col in cols if col in df.columns]
            return df.drop(columns=existing_cols)
        elif isinstance(df, SparkDataFrame):
            existing_cols = [col for col in cols if col in df.columns]
            return df.drop(*existing_cols)
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def select_columns(df: Union[pd.DataFrame, SparkDataFrame], columns: List[str]) -> Union[
        pd.DataFrame, SparkDataFrame]:

        if isinstance(df, pd.DataFrame):
            return df[columns]
        elif isinstance(df, SparkDataFrame):
            return df.select(*columns)
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def merge_dfs(left_df: Union[pd.DataFrame, SparkDataFrame],
                  right_df: Union[pd.DataFrame, SparkDataFrame],
                  on: List[str],
                  how: str = 'inner') -> Union[pd.DataFrame, SparkDataFrame]:

        if isinstance(left_df, pd.DataFrame) and isinstance(right_df, pd.DataFrame):
            return left_df.merge(right_df, on=on, how=how)
        elif isinstance(left_df, SparkDataFrame) and isinstance(right_df, SparkDataFrame):
            return left_df.join(right_df, on=on, how=how)
        else:
            raise TypeError("Unsupported DataFrame types. Both must be pandas or Spark DataFrames.")

    @staticmethod
    def join_dfs(left_df: Union[pd.DataFrame, SparkDataFrame],
                 right_df: Union[pd.DataFrame, SparkDataFrame],
                 on: List[str],
                 how: str = 'inner') -> Union[pd.DataFrame, SparkDataFrame]:

        return DataTransformationUtility.merge_dfs(left_df, right_df, on, how)

    @staticmethod
    def add_column(df: Union[pd.DataFrame, SparkDataFrame], column_name: str, values: Union[List, pd.Series]) -> Union[
        pd.DataFrame, SparkDataFrame]:

        if isinstance(df, pd.DataFrame):
            df[column_name] = values
            return df
        elif isinstance(df, SparkDataFrame):
            from pyspark.sql.functions import lit
            return df.withColumn(column_name, lit(values))
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def get_unique_values(df: Union[pd.DataFrame, SparkDataFrame], column: str) -> List:
        if isinstance(df, pd.DataFrame):
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the pandas DataFrame.")
            return df[column].dropna().unique().tolist()
        elif isinstance(df, SparkDataFrame):
            if column not in df.columns:
                raise ValueError(f"Column '{column}' does not exist in the Spark DataFrame.")
            return [row[column] for row in df.select(column).distinct().collect()]
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def drop_all_nan_rows(df: Union[pd.DataFrame, SparkDataFrame]) -> Union[pd.DataFrame, SparkDataFrame]:
        if isinstance(df, pd.DataFrame):
            return df.dropna(how='all')
        elif isinstance(df, SparkDataFrame):
            return df.na.drop(how='all')
        else:
            raise TypeError("Unsupported DataFrame type. Must be pandas or Spark DataFrame.")

    @staticmethod
    def convert_pandas_to_spark(pandas_df: pd.DataFrame) -> SparkDataFrame:
        spark_session = DataTransformationUtility.get_spark_session()

        return spark_session.createDataFrame(pandas_df)

    @staticmethod
    def get_schema_from_dataclass(data_class):
        """
        Constructs a Spark schema from a data class.
        """
        struct_fields = []

        for field in fields(data_class):
            # Determine field type
            if field.type == Optional[str]:
                field_type = StringType()
            else:
                field_type = StringType()  # Default to StringType for simplicity

            # Append StructField with nullable=True
            struct_fields.append(StructField(field.name, field_type, True))

        return StructType(struct_fields)

    @staticmethod
    def convert_list_dicts_to_df(data: List[Dict], use_spark: bool = True) -> Union[pd.DataFrame, SparkDataFrame]:
        if use_spark:
            spark_session = DataTransformationUtility.get_spark_session()

            schema = DataTransformationUtility.get_schema_from_dataclass(LLMData)
            df = spark_session.createDataFrame(data, schema=schema)

            return df
        else:
            df = pd.DataFrame(data)
            return df

    @staticmethod
    def dict_of_dicts_to_list_of_dicts(data: Dict) -> List[Dict]:
        return [value for key, value in data.items()]

    @staticmethod
    def concatenate_dfs(dfs: List[Union[pd.DataFrame, SparkDataFrame]], use_spark: bool = True) -> Union[
        pd.DataFrame, SparkDataFrame]:
        if not dfs:
            raise ValueError("The list of DataFrames is empty.")

        if use_spark:
            # Ensure all DataFrames are Spark DataFrames
            if not all(isinstance(df, SparkDataFrame) for df in dfs):
                raise TypeError("All DataFrames must be Spark DataFrames when use_spark is True.")

            # Use Spark to concatenate
            df_combined = dfs[0]
            for df in dfs[1:]:
                df_combined = df_combined.union(df)
            return df_combined

        else:
            # Ensure all DataFrames are Pandas DataFrames
            if not all(isinstance(df, pd.DataFrame) for df in dfs):
                raise TypeError("All DataFrames must be Pandas DataFrames when use_spark is False.")

            # Use Pandas to concatenate
            return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def get_base_identifier_key_map(identifier_keys) -> dict:
        key_map = {}

        for key in identifier_keys:
            parts = key.rsplit('_', 1)  # Split from the right by the last underscore
            base_key = parts[0]
            if len(parts) > 1 and parts[1].isdigit():  # Check if the last part is numeric
                if base_key not in key_map:
                    key_map[base_key] = []
                key_map[base_key].append(key)
            else:
                if key not in key_map:
                    key_map[key] = []
                key_map[key].append(key)

        return key_map

    class LLMTransformations:
        @staticmethod
        def generate_data_for_llm(df: Union[pd.DataFrame, SparkDataFrame]) -> List[dict]:
            # transforms df input data into expected list of dicts
            pass

        @staticmethod
        def transform_llm_output(data) -> Dict[str, Optional[LLMData]]:
            llm_clean_data = {}

            for key in data.keys():
                key_data = data.get(key, None)

                if key_data:
                    llm_clean_data[key] = LLMData(
                        element_type=key_data.get('element_type', None),
                        element_id=key_data.get('element_id', None),
                        element_name=key_data.get('element_name', None),
                        element_class=key_data.get('element_class', None),
                        element_text=key_data.get('element_text', None),
                        element_value=key_data.get('element_value', None),
                        element_location=key_data.get('element_location', None),
                        element_displayed=key_data.get('element_displayed', None),
                        css_selector=key_data.get('css_selector', None),
                        element_html=key_data.get('element_html', None),
                        data_attributes=key_data.get('data_attributes', None),
                    )
                else:
                    llm_clean_data[key] = None

            return llm_clean_data

        @staticmethod
        def convert_llm_response_to_list(data: Dict[str, Optional[LLMData]]) -> List[Dict]:
            llm_list_data = []

            for key, value in data.items():
                if value:
                    item_dict = value.to_dict()
                    item_dict['key'] = key
                    llm_list_data.append(item_dict)
                else:
                    # Append an empty dictionary with the key if value is None
                    llm_list_data.append({'key': key})

            return llm_list_data

    class LLMValidations:
        @staticmethod
        def get_base_identifier_key_map(identifier_keys) -> dict:
            key_map = {}

            for key in identifier_keys:
                parts = key.rsplit('_', 1)  # Split from the right by the last underscore
                base_key = parts[0]
                if len(parts) > 1 and parts[1].isdigit():  # Check if the last part is numeric
                    if base_key not in key_map:
                        key_map[base_key] = []
                    key_map[base_key].append(key)
                else:
                    if key not in key_map:
                        key_map[key] = []
                    key_map[key].append(key)

            return key_map

        @staticmethod
        def process_validation_on_llm_responses(df: pd.DataFrame, attributes_to_check: [str], cases: [int],
                                                identifier_keys: [str]) -> Union[bool, dict]:
            if len(identifier_keys) == 0:
                raise Exception("Please provide identifier_keys")

            identifier_key_validation = {}
            # PASS
            # attributes match for each identifier key
            # If one case has a populated identifier key

            # fail
            # if no attributes match
            # css selectors do not return same element

            passed = True
            for i_key in identifier_keys:
                # check if cases match for the given key
                comparison_result = df[df['key'] == i_key].groupby('case_num')[attributes_to_check].nunique().eq(
                    1).all()

                missing = []
                # If comparison_result is not empty, process the results
                if not comparison_result.empty:
                    for attr in attributes_to_check:
                        passed_result = comparison_result[attr]

                        if not passed_result:
                            missing.append(attr)
                            passed = False

                    # Store results
                    identifier_key_validation[i_key] = {
                        'mismatched_attributes': missing
                    }
                else:
                    # Handle case where no rows match the key
                    identifier_key_validation[i_key] = {
                        'mismatched_attributes': attributes_to_check[:]
                        # All attributes are considered missing if no rows match
                    }

            return passed, identifier_key_validation
