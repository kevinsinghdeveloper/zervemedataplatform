from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from data_platform.model_transforms.db.PipelineRunConfig import PipelineRunConfig
from data_platform.model_transforms.db.abstractions.ModelBase import ModelBase
from data_platform.model_transforms.db.helpers.data_class_helpers import primary_key, foreign_key


@dataclass
class PipelineActivityTracker(ModelBase):
    ID: Optional[int] = primary_key()  # Primary key field
    PipelineRunConfig_id: int = foreign_key(PipelineRunConfig, "ID")
    activity_log: dict = None
    updated_time_stamp: datetime = None


