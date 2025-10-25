from dataclasses import dataclass
from typing import Optional

from zervedataplatform.abstractions.types.models.SerpJobRequestResourceModel import SerpJobRequestResourceModel


@dataclass
class SerpJobProductDetailsRequestResourceModel(SerpJobRequestResourceModel):
    search_type: Optional[str] = None
    google_domain: Optional[str] = None
    location: Optional[str] = None
    global_location: Optional[str] = None
    language: Optional[str] = None
    product_id: str = None
    gpc_id: str = None


'''


url: https://api.valueserp.com/search?api_key=14454D83F7C444D38EBF367FD5F2F076&
q=keyword+here&location=98146%2C+Washington%2C+United+States&gl=us&
hl=en&google_domain=google.com&include_ai_overview=true&page=1&max_page=5&num=100
'''