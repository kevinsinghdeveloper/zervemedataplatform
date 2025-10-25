from dataclasses import dataclass

from model_transforms.db.abstractions.PipelineRunConfigBase import PipelineRunConfigBase


@dataclass
class PipelineRunConfig(PipelineRunConfigBase):
    ai_config: dict = None
    web_config: dict = None
    run_config: dict = None
    cloud_config: dict = None
    db_config: dict = None