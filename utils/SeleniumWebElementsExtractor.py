from dataclasses import asdict

from selenium_driverless.types.by import By

from abstractions.types.models.WebSelectors import WebSelectors

from utils.Utility import Utility
from utils.SeleniumBrowserDriverHandler import SeleniumBrowserDriverHandler


class SeleniumWebElementsExtractor:
    def __init__(self):
        self.__selenium_handler = SeleniumBrowserDriverHandler()

    async def get_web_data_extract(self, web_selectors: WebSelectors, url: str) -> dict | None:
        web_selectors_data = asdict(web_selectors)
        driver = await self.__selenium_handler.connect_to_url(url=url, headless=True)

        if driver is None:
            Utility.error_log(f"Failed to connect to {url}")
            return None

        web_extract = {
            'url': url
        }
        try:
            for key in web_selectors_data:
                selectors = web_selectors_data.get(key, [])

                if not selectors:
                    Utility.warning_log(f"Missing selector for {key}")
                    continue

                extracts = []
                for s in selectors:
                    if s and s != '':
                        element = await driver.find_elements(by=By.CSS_SELECTOR, value=s)
                        text = await SeleniumWebElementsExtractor.get_element_text(element)
                        extracts.append(text)

                web_extract[key] = extracts

        except Exception as e:
            Utility.error_log(f"Extraction failed for {url}: {e}")
            return None

        finally:
            if driver:
                await driver.quit()

        # Utility.log(f"Found elements {str(web_extract)}")
        return web_extract

    @staticmethod
    async def get_element_text(element):
        text = None
        # For a single element (like your original case)
        if len(element) == 1:
            return await element[0].text

        # For multiple elements (like li in a ul)
        texts = []
        for el in element:
            text = await el.text
            texts.append(text)

        # Join with space as separator
        return ", ".join(texts)

    @staticmethod
    def get_web_selectors(config: dict):
        # convert config to WebSelector
        web_selector = WebSelectors(
            title=config.get("title", []),
            product_details_summary = config.get("product_details_summary", []),
            product_details_vertical = config.get("product_details_vertical", []),
            product_details_list_horizontal = config.get("product_details_list_horizontal", []),
            review_traits = config.get("review_traits", []),
            size_variations = config.get("size_variations", []),
            color_variations = config.get("color_variations", []),
            alternate_sites = config.get("alternate_sites", [])
        )

        return web_selector