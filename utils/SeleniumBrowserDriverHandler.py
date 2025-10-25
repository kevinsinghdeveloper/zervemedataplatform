from selenium_driverless import webdriver

import asyncio

from selenium.common.exceptions import (
    WebDriverException as SeleniumWebDriverException,
    TimeoutException as SeleniumTimeoutException,
    InvalidArgumentException as SeleniumInvalidArgumentException, TimeoutException, InvalidArgumentException,
    WebDriverException
)

from data_platform.utils.Utility import Utility

# Define the browser options
driver_options_mapping = {
    'chrome': webdriver.ChromeOptions()
}
PAGE_TIMEOUT_DEFAULT = 200
# Add headless options to Chrome and Firefox
# TODO move to web_config

# driver_options_mapping['chrome'].binary_location = "/usr/bin/chromium"

driver_options_mapping['chrome'].add_argument('--disable-gpu')
driver_options_mapping['chrome'].add_argument('--disable-dev-shm-usage')
driver_options_mapping['chrome'].add_argument('--disable-extensions')
driver_options_mapping['chrome'].add_argument("--disable-javascript")
driver_options_mapping['chrome'].add_argument('--disable-popup-blocking')
# driver_options_mapping['chrome'].add_argument('--no-sandbox') # ONLY for docker
driver_options_mapping['chrome'].add_argument('--disable-notifications')
driver_options_mapping['chrome'].add_argument(
    '--disable-infobars')  # Disable info bars (e.g., "Chrome is being controlled by automated software")

class SeleniumBrowserDriverHandler:
    @staticmethod
    async def get_driver_for_browser(browser_type: str, headless: bool = False):
        if headless:
            driver_options_mapping['chrome'].add_argument('--headless')

        options = driver_options_mapping.get(browser_type.lower(), None)

        if browser_type.lower() == 'chrome':
            return await webdriver.Chrome(options=options)
        else:
            return None

    @staticmethod
    async def connect_to_url(url: str, browser: str = 'chrome', implicitly_wait: int = 1, headless=False):
        try:
            driver = await SeleniumBrowserDriverHandler.get_driver_for_browser(browser, headless)

            # more hidden stuff
            await driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            await driver.get(url, wait_load=True)

            await asyncio.sleep(implicitly_wait)

        except (TimeoutException, SeleniumTimeoutException) as e:

            Utility.error_log(f"Timeout error: {str(e)}")
            raise

        except (InvalidArgumentException, SeleniumInvalidArgumentException) as e:

            Utility.error_log(f"Invalid argument: {str(e)}")
            raise

        except (WebDriverException, SeleniumWebDriverException) as e:

            Utility.error_log(f"WebDriver error: {str(e)}")
            raise

        except Exception as ex:

            Utility.error_log(f"Unexpected error: {str(ex)}")
            raise

        return driver

