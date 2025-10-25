from dataclasses import dataclass

from data_platform.model_transforms.db.abstractions.SitesBase import SitesBase


@dataclass
class Sites(SitesBase):
    name: str = None
    url: str = None


