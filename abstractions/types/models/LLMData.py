from dataclasses import dataclass, field, fields
from typing import Optional, Dict, List


@dataclass
class LLMData:
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

    def to_dict(self):
        return {
            'element_type': self.element_type,
            'element_id': self.element_id,
            'element_name': self.element_name,
            'element_class': self.element_class,
            'element_text': self.element_text,
            'element_value': self.element_value,
            'element_location': self.element_location,
            'element_displayed': self.element_displayed,
            'css_selector': self.css_selector,
            'element_html': self.element_html,
            'data_attributes': self.data_attributes,
        }

    @staticmethod
    def get_all_fields() -> List[str]:
        return [field.name for field in fields(LLMData)]

