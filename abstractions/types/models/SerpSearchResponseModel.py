from dataclasses import dataclass
from typing import Optional

@dataclass
class SerpSearchResponseModel:
    product_title: Optional[str] = None
    product_id: Optional[str] = None
    gpc_id: Optional[str] = None
    url: Optional[str] = None
    merchant: Optional[str] = None
    price: Optional[float] = None
    position_rank: Optional[int] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    product_image: Optional[str] = None