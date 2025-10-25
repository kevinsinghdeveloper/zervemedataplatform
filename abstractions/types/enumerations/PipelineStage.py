from enum import Enum


class PipelineStage(Enum):
    initialize_task = 0
    pre_validate_task = 1
    read_task = 2
    main_task = 3
    output_task = 4
