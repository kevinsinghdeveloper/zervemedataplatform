from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class WebSelectors:
    title: [str] = None
    product_details_summary: [str] = None
    product_details_vertical: [str] = None
    product_details_list_horizontal: [str] = None
    review_traits: [str] = None
    size_variations: [str] = None
    color_variations: [str] = None
    alternate_sites: [str] = None