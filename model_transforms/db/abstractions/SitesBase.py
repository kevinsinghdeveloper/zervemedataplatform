from dataclasses import dataclass
from typing import Optional

from data_platform.model_transforms.db.PipelineRunConfig import PipelineRunConfig
from data_platform.model_transforms.db.abstractions.ModelBase import ModelBase
from data_platform.model_transforms.db.helpers.data_class_helpers import primary_key, foreign_key


@dataclass
class SitesBase(ModelBase):
    ID: Optional[int] = primary_key()  # Primary key field
    UpdatedPipelineRunConfig_id: int = foreign_key(PipelineRunConfig, "ID")