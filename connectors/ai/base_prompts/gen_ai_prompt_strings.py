LLM_INSTRUCTIONS_FOR_PRODUCT = """
When conducting your analysis please follow this order when reviewing each attribute for a particular case:
1. data attributes (data-*)
2. id (e.g., product-name, tax-order, etc.) class name
3. class name
4. name
5. text
6. value

Ensure the data includes a CSS SELECTOR.

Please identify and return one of each of the following elements:

1. A button to add the product to a cart, with straightforward text like "add to cart" or "add to basket" ONLY (key=add_to_cart).
2. A button to buy the product now ONLY (key=buy_now).
3. The quantity of the product, represented as a select list for the number of items ONLY. (key=quantity). 
4. The name of the product name ONLY, do not include brand. subtitle, title, etc... ONLY the product name! (key=product_name).

Do not derive data. DO NOT generate CSS selectors, use the CSS selectors provided in data. Instead, pull the data as is and manipulate the structure based on these instructions. 

Return the results as a dict of dicts, with the keys specified above. Include all fields in the results.

If any keys return an empty result, return an empty dict. All fields should be double-quoted, as the output will be passed into a `json.loads` call.

**Example of one item:**

{{
    "quantity" : {{
                "element_id": "qty_1785856",
                "element_name": "",
                "element_class": "css-1hmuo0f",
                "element_text": "1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10",
                "element_value": "1",
                "element_location": "{{{{'x': 16, 'y': 5708}}}}",
                "element_displayed": "True",
                "element_html": ""<button class=\"accordion-header accordion-header-button heavy grid-x c-capitalize\" aria-expanded=\"true\">"",
                "css_selector": "select[id=\\"qty_1785856\\"][class=\\"css-1hmuo0f\\"]"
        }}
}}
"""


LLM_INSTRUCTIONS_FOR_CART = """
When conducting your analysis please follow this order when reviewing each attribute for a particular case:
1. data attributes (data-*)
2. id (e.g., product-name, tax-order, etc.) class name
3. class name
4. name
5. text
6. value

Ensure the data includes a CSS SELECTOR.

Please identify and return one of each of the following elements:

1. An item or element representing a cart/bag/basket item. (key=cart_item)
2. An item or element containing all cart/bag/basket items. (key=cart)
3. The pay/purchase/buy now button or checkout button, which might be a different element like a link. This can be multiple, and `<n>` will increment for each case. (key=pay_button_<n>)
4. A button or element to save an item for later. (key=save_for_later_button)
5. The quantity of a product, typically a select list for the number of items. (key=quantity_button)
6. The selected quantity value of a product, not the list, but the chosen value. (key=quantity)
7. The price of a product within the cart item. (key=price)
8. The name of the product (e.g., product name or title). If a subtitle is available, please include it. (key=name)
9. The URL of a product, typically the link on a cart item that redirects to the product. (key=url)

Do not derive data. DO NOT generate CSS selectors, use the CSS selectors provided in data Instead, pull the data as is and manipulate the structure based on these instructions. 

Return the results as a dict of dicts, with the keys specified above. Include all fields in the results.

If any keys return an empty result, return an empty dict. All fields should be double-quoted, as the output will be passed into a `json.loads` call.

**Example of one item:**

{{
    "quantity" : {{
                "element_id": "qty_1785856",
                "element_name": "",
                "element_class": "css-1hmuo0f",
                "element_text": "1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10",
                "element_value": "1",
                "element_location": "{{{{'x': 16, 'y': 5708}}}}",
                "element_displayed": "True",
                "element_html": ""<button class=\"accordion-header accordion-header-button heavy grid-x c-capitalize\" aria-expanded=\"true\">"",
                "css_selector": "select[id=\\"qty_1785856\\"][class=\\"css-1hmuo0f\\"]"
        }}
}}
"""

LLM_INSTRUCTIONS_FOR_CHECKOUT = """
When conducting your analysis please follow this order when reviewing each attribute for a particular case:
1. data attributes (data-*)
2. id (e.g., product-name, tax-order, etc.) class name
3. class name
4. name
5. text
6. value

Ensure the data includes a CSS SELECTOR.

Please identify and return one of each of the following elements:

1. The total amount or sum of all products, representing the checkout total value or subtotal amount. (key=total_amount) - Return only one.
2. An element containing the shipping cost. (key=shipping) - Return only one.
3. The element containing the order tax value. (key=tax) - Return only one.
4. A checkout/purchase/order now button or element to buy the product now. This might be a different element like a link. (key=pay_now)

Do not derive data. DO NOT generate CSS selectors, use the CSS selectors provided in data. Instead, pull the data as is and manipulate the structure based on these instructions. 

Return the results as a dict of dicts, with the keys specified above. Include all fields in the results.

If any keys return an empty result, return an empty dict. All fields should be double-quoted, as the output will be passed into a `json.loads` call.

**Example of one item:**
{{
    "quantity" : {{
                "element_id": "qty_1785856",
                "element_name": "",
                "element_class": "css-1hmuo0f",
                "element_text": "1\\n2\\n3\\n4\\n5\\n6\\n7\\n8\\n9\\n10",
                "element_value": "1",
                "element_location": "{{{{'x': 16, 'y': 5708}}}}",
                "element_displayed": "True",
                "element_html": ""<button class=\"accordion-header accordion-header-button heavy grid-x c-capitalize\" aria-expanded=\"true\">"",
                "css_selector": "select[id=\\"qty_1785856\\"][class=\\"css-1hmuo0f\\"]"
        }}
}}
"""


LLM_INSTRUCTIONS_FOR_LOGIN = """
When conducting your analysis please follow this order when reviewing each attribute for a particular case:
1. data attributes (data-*)
2. id (e.g., product-name, tax-order, etc.) class name
3. class name
4. name
5. text
6. value

Ensure the data includes a CSS SELECTOR.

Please identify and return one of each of the following elements:

1. A text input field for the username or email. (key=username)
2. A text input field for the password. (key=password)
3. An element (could be a button or another element) to continue to the next step. (key=continue_button)
4. An element (could be a button or another element) to log in. (key=login_button)

Do not derive data. DO NOT generate CSS selectors, use the CSS selectors provided in data. Instead, pull the data as is and manipulate the structure based on these instructions. 

Return the results as a dict of dicts, with the keys specified above. Include all fields in the results.

If any keys return an empty result, return an empty dict. All fields should be double-quoted, as the output will be passed into a `json.loads` call.

**Example of one item:**

```json
{{
  "username": {{
        "element_id": "ap_email",
        "element_name": "email",
        "element_class": "a-input-text a-span12 auth-autofocus auth-required-field auth-require-claim-validation",
        "element_text": "",
        "element_value": "",
        "element_location": "{{{{'x': 16, 'y': 5708}}}}",
        "element_displayed": true,
        "element_html": ""<button class=\"accordion-header accordion-header-button heavy grid-x c-capitalize\" aria-expanded=\"true\">"",
        "css_selector": "input[id='ap_email'][name='email']"
  }}
}}
"""

LLM_INSTRUCTIONS_FOR_DETERMINING_LOGIN_CASE = """
Goal: Identify the type of login page, by looking at a list of elements and their corresponding attributes.

Cases to check, needs to follow all rules under each case to be identified as a particular case:

Standard Login Page:
    Contains username or email, password, and a submit or normal button.
    Not a modal or popup.
    Return: 1
    
Username/Email Only Login Page:
    Contains only username or email and a submit or normal button. No password field.
    Not a modal or popup
    Return: 2
    
Modal or Popup Login Page:
    Contains username or email, password, and a submit or normal button.
    Presented as a modal or popup.
    Return: 3

None:
    return 0

Please return a single int value ONLY
"""