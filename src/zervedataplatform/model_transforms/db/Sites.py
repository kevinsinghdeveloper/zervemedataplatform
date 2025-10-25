from dataclasses import dataclass

from zervedataplatform.model_transforms.db.abstractions.SitesBase import SitesBase


@dataclass
class Sites(SitesBase):
    name: str = None
    url: str = None


