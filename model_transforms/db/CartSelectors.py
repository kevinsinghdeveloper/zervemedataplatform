from dataclasses import dataclass

from data_platform.model_transforms.db.abstractions.SelectorsBase import SelectorsBase


@dataclass
class CartSelectors(SelectorsBase):
    cart_item: str = None
    cart: str = None
    pay_button: str = None
    save_for_later_button: str = None
    quantity_button: str = None
    quantity: str = None
    price: str = None
    name: str = None
    url: str = None

