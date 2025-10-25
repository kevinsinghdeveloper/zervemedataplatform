from dataclasses import dataclass
from typing import Optional

@dataclass
class SerpProductDetailsResponseModel:
    product_id: Optional[str] = None
    gpc_id: Optional[str] = None
    description: Optional[str] = None
    sellers: Optional[str] = None
    specs: Optional[str] = None