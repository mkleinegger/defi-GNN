"""DeFi service classification with GraphSAGE."""

from .config import DataConfig, ProjectConfig, TrainingConfig, load_project_config

__all__ = [
    "DataConfig",
    "ProjectConfig",
    "TrainingConfig",
    "load_project_config",
]

__version__ = "0.1.0"
