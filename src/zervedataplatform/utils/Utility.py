import json
import logging
import os
import io
from dataclasses import is_dataclass, asdict, fields
from datetime import datetime
from typing import Dict, Any, List, TypeVar, Type
import re
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, Row

T = TypeVar("T")
class Utility:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set default logging level to DEBUG

    if not logger.hasHandlers():
        # Create console handler and set level to INFO
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create file handler and set level to DEBUG
        os.makedirs('logs', exist_ok=True)
        log_filename = f"logs/app_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Add formatter to handlers
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    @staticmethod
    def log(message: str) -> None:
        Utility.logger.info(message)

    @staticmethod
    def debug_log(message: str) -> None:
        Utility.logger.debug(message)

    @staticmethod
    def warning_log(message: str) -> None:
        Utility.logger.warning(message)

    @staticmethod
    def error_log(message: str) -> None:
        Utility.logger.error(message)

    @staticmethod
    def critical_log(message: str) -> None:
        Utility.logger.critical(message)

    @staticmethod
    def read_in_json_file(json_path: str) -> Dict[str, Any]:
        try:
            with open(json_path, 'r') as file:
                content = file.read()
                config = json.loads(content)
                return config
        except json.JSONDecodeError as e:
            Utility.error_log(f"JSONDecodeError: {e.msg} at line {e.lineno} column {e.colno}")
            Utility.error_log(f"Problematic JSON content: {content}")
        except FileNotFoundError:
            Utility.error_log(f"File not found: {json_path}")
            return None
        except Exception as e:
            Utility.error_log(f"An error occurred: {e}")
        raise

    @staticmethod
    def write_dict_to_json_file(data: dict, file_path: str):
        try:
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            Utility.log(f"Dictionary successfully written to {file_path}")
        except Exception as e:
            Utility.error_log(f"An error occurred while writing to the JSON file: {e}")

    @staticmethod
    def read_image(image_path: str) -> bytes:
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()
                Utility.log(f"Image read successfully from {image_path}.")
                return content
        except FileNotFoundError:
            Utility.error_log(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            Utility.error_log(f"Failed to read image file: {e}")
            raise

    @staticmethod
    def trim_spaces(input_string):
        # Split the string into words and join them with a single space
        trimmed_string = ' '.join(input_string.split())
        return trimmed_string

    @staticmethod
    def clean_and_convert(value: any) -> float | int:
        # Handle None case first
        if value is None:
            return 0

        if type(value) in [int, float]:
            return value

        # Use regex to keep only digits, a single dot, and an optional leading minus sign
        cleaned_value = re.sub(r"[^\d.-]", "", str(value))  # Added str() conversion

        # Handle edge cases where the string is empty or invalid (e.g., just a dot/minus sign)
        if not cleaned_value or cleaned_value in {"-", "."}:
            return 0

        try:
            # Convert to int if there's no decimal point, otherwise convert to float
            return int(cleaned_value) if "." not in cleaned_value else float(cleaned_value)
        except ValueError:
            # Return 0 if the conversion fails
            return 0

    @staticmethod
    def dataclass_to_dataframe(dataclass_list: List[T], return_spark: bool = False, spark: SparkSession = None):
        """
        Converts a list of dataclass instances to a Pandas or Spark DataFrame.

        Args:
            dataclass_list (List[T]): List of dataclass instances.
            return_spark (bool): If True, returns a Spark DataFrame. Otherwise, returns a Pandas DataFrame.
            spark (SparkSession, optional): Required if returning a Spark DataFrame.

        Returns:
            pd.DataFrame | pyspark.sql.DataFrame: A DataFrame in Pandas or Spark format.
        """
        if not dataclass_list:
            return pd.DataFrame() if not return_spark else spark.createDataFrame([], schema=StructType())

        if not is_dataclass(dataclass_list[0]):
            raise ValueError("The provided list must contain dataclass instances.")

        # Convert dataclass list to a list of dictionaries
        dict_list = [asdict(item) for item in dataclass_list]

        # ✅ Return Pandas DataFrame
        if not return_spark:
            return pd.DataFrame(dict_list)

        # ✅ Return Spark DataFrame (convert dicts to Rows)
        if spark is None:
            raise ValueError("A SparkSession is required to return a Spark DataFrame.")

        # Dynamically create StructType schema from dataclass fields
        field_mappings = {
            int: IntegerType(),
            float: DoubleType(),
            str: StringType(),
        }

        schema = StructType([
            StructField(field.name, field_mappings.get(field.type, StringType()), True)
            for field in fields(dataclass_list[0])
        ])

        # Convert dictionaries to Row objects
        row_list = [Row(**data) for data in dict_list]

        return spark.createDataFrame(row_list, schema=schema)

    @staticmethod
    def dataframe_to_dataclass(df: pd.DataFrame, dataclass_type: Type[T]) -> List[T]:
        """
        Converts a Pandas DataFrame to a list of dataclass instances.

        Args:
            df (pd.DataFrame): The DataFrame to convert.
            dataclass_type (Type[T]): The dataclass type to convert each row into.

        Returns:
            List[T]: A list of dataclass instances.
        """
        if not is_dataclass(dataclass_type):
            raise ValueError("The provided type must be a dataclass.")

        try:
            return [dataclass_type(**row.to_dict()) for _, row in df.iterrows()]
        except Exception as e:
            raise ValueError(f"Error converting DataFrame to dataclass list: {e}")
