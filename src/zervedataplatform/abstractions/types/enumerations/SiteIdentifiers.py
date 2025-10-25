from enum import Enum

# Also controls the order in which the identifiers are processed


class SiteIdentifiers(Enum):
    product = 'product'
    login = 'login'
    checkout = 'checkout'
    cart = 'cart'