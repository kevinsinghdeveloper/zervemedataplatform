from dataclasses import dataclass
from typing import Optional

from zervedataplatform.model_transforms.db.PipelineRunConfig import PipelineRunConfig
from zervedataplatform.model_transforms.db.abstractions.ModelBase import ModelBase
from zervedataplatform.model_transforms.db.helpers.data_class_helpers import primary_key, foreign_key


@dataclass
class SitesBase(ModelBase):
    ID: Optional[int] = primary_key()  # Primary key field
    UpdatedPipelineRunConfig_id: int = foreign_key(PipelineRunConfig, "ID")