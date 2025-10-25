"""Pipeline orchestration and execution framework."""

# Avoid circular imports - import on demand
__all__ = [
    "PipelineUtility",
    "FuncDataPipe",
    "DataConnectorBase",
    "DataPipeline",
]

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name in __all__:
        from .Pipeline import (
            PipelineUtility,
            FuncDataPipe,
            DataConnectorBase,
            DataPipeline,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
