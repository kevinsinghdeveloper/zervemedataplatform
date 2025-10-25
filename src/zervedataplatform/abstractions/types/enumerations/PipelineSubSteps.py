from enum import Enum


class PipelineSubSteps(Enum):
    Extract = "extract"
    Transform = "transform"
    DataRequest = "request"
    Validation = 'validation'