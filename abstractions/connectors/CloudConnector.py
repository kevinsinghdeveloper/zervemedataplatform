from typing import Dict, List

import pandas as pd

from abc import abstractmethod, ABC


class CloudConnector(ABC):
    def __init__(self, cloudConfig: dict = None):
        self._config = cloudConfig

    @abstractmethod
    def get_dataframe_from_cloud(self, file_path: str, sep: str="|") -> pd.DataFrame:
        """Fetches a file from the cloud and returns it as a pandas DataFrame."""
        pass

    @abstractmethod
    def upload_data_frame_to_cloud(self, df: pd.DataFrame, file_path: str, sep: str="|") -> None:
        """Uploads a pandas DataFrame to the cloud as a CSV file."""
        pass

    @abstractmethod
    def get_dict_from_cloud(self, file_path: str) -> Dict:
        """Fetches a JSON file from the cloud and returns it as a Python dict."""
        pass

    @abstractmethod
    def upload_dict_to_cloud(self, data: Dict, file_path: str) -> None:
        """Uploads a Python dictionary to the cloud as a JSON file."""
        pass

    @abstractmethod
    def list_files(self, prefix: str, item_type: str = 'all') -> List[str]:
        pass

    @abstractmethod
    def move_folder(self, source_path: str, destination_path: str) -> bool:
        pass

    @abstractmethod
    def copy_folder(self, source_path: str, destination_path: str) -> bool:
        pass

