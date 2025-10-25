from abstractions.types.enumerations.SiteIdentifiers import SiteIdentifiers
from model_transforms.db.ProductSelectors import product_selectors_instance


def generate_output(product_selector_map):
    output_lines = []
    for i, (key, description) in enumerate(product_selector_map.items(), start=1):
        output_lines.append(f"{i}. {description} (key={key})")
    return "\n".join(output_lines)


class BasePromptHandler:

    @staticmethod
    def generate_base_prompt(identifier: SiteIdentifiers):

        if identifier == SiteIdentifiers.product:
            key_llm_desc = generate_output(product_selectors_instance)

            return f"""
                    When conducting your analysis please follow this order when reviewing each attribute for a particular case:
                    1. data attributes (data-*)
                    2. id (e.g., product-name, tax-order, etc.) class name
                    3. class name
                    4. name
                    5. text
                    6. value
                    
                    Ensure the data includes a CSS SELECTOR.
                    
                    Please identify and return one of each of the following elements:
                    
                    {key_llm_desc}
                    
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