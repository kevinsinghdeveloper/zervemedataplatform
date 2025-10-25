import json
import os
from io import StringIO, BytesIO
from typing import Dict, List
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError, BotoCoreError

from zervedataplatform.abstractions.connectors.CloudConnector import CloudConnector
from zervedataplatform.utils.Utility import Utility

CSV_DEL = "|"
class S3CloudConnector(CloudConnector):
    def __init__(self, cloudConfig: dict = None):
        super().__init__(cloudConfig)

        # Use environment variables if cloudConfig is None
        if cloudConfig is None:
            cloudConfig = {
                'bucket': os.environ.get('S3_BUCKET'),
                'key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
                'secret_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
                'region': os.environ.get('AWS_REGION', 'us-west-2'),  # Default to us-east-1 if not set
                'settings': {}
            }

        self.bucket = cloudConfig['bucket']

        self.settings = cloudConfig.get('settings', {})

        self.s3_session = boto3.Session(
            aws_access_key_id=cloudConfig['key_id'],
            aws_secret_access_key=cloudConfig['secret_key'],
            region_name=cloudConfig['region']
        )

    def get_dataframe_from_cloud(self, file_path, sep: str="|") -> pd.DataFrame:
        """
        Fetch CSV data from S3 and convert it to a pandas DataFrame.

        Args:
            file_path (str): Path to the CSV file in S3

        Returns:
            pd.DataFrame: DataFrame containing the data, or empty DataFrame if error occurs
        """
        try:
            s3 = self.s3_session.client("s3")
            obj = s3.get_object(Bucket=self.bucket, Key=file_path)

            # Read CSV directly from the S3 object using StringIO
            data = obj['Body'].read().decode('utf-8')
            return pd.read_csv(StringIO(data), sep=CSV_DEL)

        except NoCredentialsError:
            print("Credentials not available for S3 access.")
        except PartialCredentialsError:
            print("Incomplete credentials provided for S3 access.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                print(f"Bucket {self.bucket} does not exist.")
            elif error_code == 'AccessDenied':
                print(f"Access denied for bucket {self.bucket}. Check bucket policies.")
            elif error_code == 'InvalidAccessKeyId':
                print("The AWS Access Key ID is invalid.")
            elif error_code == 'SignatureDoesNotMatch':
                print("The request signature we calculated does not match the signature you provided.")
            else:
                print(f"Error fetching CSV from S3: {e}")
        except pd.errors.EmptyDataError:
            print("The CSV file is empty.")
        except BotoCoreError as e:
            print(f"BotoCoreError encountered: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

        # Return empty DataFrame on error
        return pd.DataFrame()

    def upload_data_frame_to_cloud(self, df: pd.DataFrame, file_path: str, sep: str="|") -> None:
        """
        Uploads a Pandas DataFrame as either CSV or Parquet to an S3 bucket.

        :param df: The Pandas DataFrame to be uploaded.
        :param file_path: The target path in the S3 bucket. File extension determines format:
                         .csv for CSV, .parquet for Parquet.
        """
        try:
            file_extension = file_path.lower().split('.')[-1]
            s3 = self.s3_session.client("s3")

            if file_extension == 'csv':
                # Convert DataFrame to CSV
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False, sep=CSV_DEL)
                file_content = csv_buffer.getvalue()

            elif file_extension == 'parquet':
                # Convert DataFrame to Parquet
                parquet_buffer = BytesIO()
                df.to_parquet(parquet_buffer, engine='pyarrow', index=False)
                parquet_buffer.seek(0)
                file_content = parquet_buffer.getvalue()

            else:
                raise ValueError(f"Unsupported file format: {file_extension}. Use .csv or .parquet")

            # Upload to S3
            s3.put_object(Bucket=self.bucket, Key=file_path, Body=file_content)
            Utility.log(f"File uploaded successfully to {file_path}")

        except NoCredentialsError:
            Utility.error_log("Credentials not available for S3 access.")
        except PartialCredentialsError:
            Utility.error_log("Incomplete credentials provided for S3 access.")
        except ClientError as e:
            Utility.error_log(f"Client error occurred: {e}")
        except BotoCoreError as e:
            Utility.error_log(f"BotoCoreError encountered: {e}")
        except Exception as e:
            Utility.error_log(f"Error uploading data to S3: {e}")

    def get_dict_from_cloud(self, file_path: str) -> Dict:
        """
        Reads a JSON file from S3 and converts it into a Python dictionary.
        :param file_path: The path of the JSON file in the S3 bucket.
        :return: A Python dictionary containing the data from the JSON file.
        """
        try:
            # Fetch file from S3
            s3 = self.s3_session.client("s3")
            obj = s3.get_object(Bucket=self.bucket, Key=file_path)
            data = obj['Body'].read().decode('utf-8')
            json_data = json.loads(data)
            return json_data
        except NoCredentialsError:
            print("Credentials not available for S3 access.")
        except PartialCredentialsError:
            print("Incomplete credentials provided for S3 access.")
        except ClientError as e:
            print(f"Client error occurred: {e}")
        except json.JSONDecodeError:
            print("Error decoding JSON data.")
        except BotoCoreError as e:
            print(f"BotoCoreError encountered: {e}")
        except Exception as e:
            print(f"Error fetching JSON from S3: {e}")
        return {}

    def upload_dict_to_cloud(self, data: Dict, file_path: str) -> None:
        """
        Uploads a Python dictionary as a JSON file to an S3 bucket.
        :param data: The Python dictionary to be uploaded.
        :param file_path: The target path in the S3 bucket.
        """
        try:
            # Convert dict to JSON string
            json_data = json.dumps(data)

            # Upload to S3
            s3 = self.s3_session.client("s3")
            s3.put_object(Bucket=self.bucket, Key=file_path, Body=json_data)
            print(f"JSON file uploaded successfully to {file_path}")
        except NoCredentialsError:
            print("Credentials not available for S3 access.")
        except PartialCredentialsError:
            print("Incomplete credentials provided for S3 access.")
        except ClientError as e:
            print(f"Client error occurred: {e}")
        except BotoCoreError as e:
            print(f"BotoCoreError encountered: {e}")
        except Exception as e:
            print(f"Error uploading JSON to S3: {e}")

    # def list_files(self, prefix: str) -> List[str]:
    #     """List files in a specific S3 bucket folder."""
    #     s3_client = self.s3_session.client('s3')
    #     try:
    #         response = s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
    #         return [item['Key'] for item in response.get('Contents', [])]
    #     except ClientError as e:
    #         print(f"Error listing files in S3: {e}")
    #         return []

    def list_files(self, prefix: str, item_type: str = 'all') -> List[str]:
        """
        List files or folders in a specific S3 bucket folder.

        Args:
            prefix (str): The S3 prefix (folder path) to list contents from
            item_type (str): Type of items to return - 'files', 'folders', or 'all' (default)

        Returns:
            List[str]: List of file/folder paths
        """
        s3_client = self.s3_session.client('s3')
        try:
            # Ensure prefix ends with '/' if not empty
            if prefix and not prefix.endswith('/'):
                prefix += '/'

            response = s3_client.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter='/')
            items = []

            # Handle folders (CommonPrefixes)
            if item_type in ['folders', 'all']:
                folders = response.get('CommonPrefixes', [])
                items.extend(prefix['Prefix'] for prefix in folders)

            # Handle files
            if item_type in ['files', 'all']:
                files = response.get('Contents', [])
                items.extend(
                    item['Key'] for item in files
                    if item['Key'] != prefix  # Exclude the prefix itself
                    and not item['Key'].endswith('/')  # Exclude folder markers
                )

            return items

        except ClientError as e:
            print(f"Error listing items in S3: {e}")
            return []

