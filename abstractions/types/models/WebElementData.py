from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class WebElementData:
    URL: Optional[str] = None
    Browser: Optional[str] = None
    unique_id: Optional[str] = None
    element_type: Optional[str] = None
    element_displayed: Optional[bool] = None
    element_class: Optional[str] = None
    element_name: Optional[str] = None
    element_value: Optional[str] = None
    element_text: Optional[str] = None
    element_id: Optional[str] = None
    element_location: Optional[str] = None
    element_html: Optional[str] = None
    css_selector: Optional[str] = None
    data_attributes: Dict = field(default_factory=dict)
