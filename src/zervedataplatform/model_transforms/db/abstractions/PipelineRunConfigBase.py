from dataclasses import dataclass
from typing import Optional

from zervedataplatform.model_transforms.db.abstractions.ModelBase import ModelBase
from zervedataplatform.model_transforms.db.helpers.data_class_helpers import primary_key


@dataclass
class PipelineRunConfigBase(ModelBase):
    ID: Optional[int] = primary_key()  # Primary key field
    pipeline_run_id: Optional[str] = None

