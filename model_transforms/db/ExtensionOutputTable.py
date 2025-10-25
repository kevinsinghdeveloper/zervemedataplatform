from dataclasses import dataclass
from typing import Optional

from model_transforms.db.abstractions.ModelBase import ModelBase
from model_transforms.db.helpers.data_class_helpers import primary_key


@dataclass
class ExtensionOutputTable(ModelBase):
    ID: Optional[int] = primary_key()  # Primary key field
    data: dict = None
