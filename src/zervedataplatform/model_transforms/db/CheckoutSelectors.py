from dataclasses import dataclass

from zervedataplatform.model_transforms.db.abstractions.SelectorsBase import SelectorsBase


@dataclass
class CheckoutSelectors(SelectorsBase):
    total_amount: str = None
    shipping: str = None
    tax: str = None
    pay_now: str = None

