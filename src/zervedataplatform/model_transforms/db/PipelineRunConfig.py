from dataclasses import dataclass
from typing import Optional

from zervedataplatform.model_transforms.db.abstractions.PipelineRunConfigBase import PipelineRunConfigBase


@dataclass
class PipelineRunConfig(PipelineRunConfigBase):
    ai_config: dict = None
    ai_embeddings_config: Optional[dict] = None
    web_config: dict = None
    run_config: dict = None
    cloud_config: dict = None
    db_config: dict = None
    dest_db_config: Optional[dict] = None