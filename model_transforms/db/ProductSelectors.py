from dataclasses import dataclass

from data_platform.model_transforms.db.abstractions.SelectorsBase import SelectorsBase


@dataclass
class ProductSelectors(SelectorsBase):
    add_to_cart: str = None
    buy_now: str = None
    quantity: str = None
    product_name: str = None


