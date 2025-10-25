from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMProductRequestData:
    super_category: str
    product_title: str
    product_information: Optional[str] = None


@dataclass
class LLMCharacteristicOption:
    characteristic: str
    description: str
    is_multi: bool
    options: list


