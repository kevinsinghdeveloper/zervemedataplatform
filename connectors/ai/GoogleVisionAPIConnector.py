import os
from google.cloud.vision_v1 import types
from typing import List, Tuple, Optional

import cv2
import numpy as np
from google.cloud import vision

from data_platform.abstractions.connectors.AIApiConnectorBase import AiApiConnectorBase
from abstractions.shapes.Rectangle import Rectangle
from data_platform.utils.Utility import Utility


class GoogleVisionAPIConnector(AiApiConnectorBase):

    def __init__(self, ai_api_config_path: str):
        super().__init__(ai_api_config_path)

        self.__client = None

    def initialize_api(self) -> None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self._config_path

        try:
            self.__client = vision.ImageAnnotatorClient()
            Utility.log("Google Vision API client initialized successfully.")
        except Exception as e:
            Utility.error_log(f"Failed to initialize Google Vision API client: {e}")
            raise

    # TODO make modular
    def detect_text(self, image_content: bytes) -> List[types.EntityAnnotation]:
        try:
            image = vision.Image(content=image_content)
            response = self.__client.text_detection(image=image)

            if response.error.message:
                raise Exception(f"API Error: {response.error.message}")

            Utility.log("Text detection completed successfully.")
            return response.text_annotations
        except Exception as e:
            Utility.error_log(f"Text detection failed: {e}")
            raise

    def find_add_to_cart(self, texts: List[types.EntityAnnotation]) -> Tuple[int, int, int, int]:
        try:
            # Define the increased detection radius
            horizontal_margin = 70  # Increased from 70
            vertical_margin = 10     # Increased from 10

            for i in range(1, len(texts) - 2):
                if "Add" in texts[i].description and "to" in texts[i + 1].description and "Cart" in texts[i + 2].description:
                    Utility.log(f"'Add to Cart' button found in text annotations.")

                    # Collect vertices from the detected text annotations
                    vertices = [(vertex.x, vertex.y) for vertex in texts[i].bounding_poly.vertices]
                    vertices.extend([(vertex.x, vertex.y) for vertex in texts[i + 1].bounding_poly.vertices])
                    vertices.extend([(vertex.x, vertex.y) for vertex in texts[i + 2].bounding_poly.vertices])

                    # Calculate the bounding box with the increased detection radius
                    min_x = min(vertex[0] for vertex in vertices) - horizontal_margin
                    max_x = max(vertex[0] for vertex in vertices) + horizontal_margin
                    min_y = min(vertex[1] for vertex in vertices) - vertical_margin
                    max_y = max(vertex[1] for vertex in vertices) + vertical_margin

                    return min_x, min_y, max_x - min_x, max_y - min_y

            Utility.log("'Add to Cart' button not found in text annotations.")
            raise ValueError("'Add to Cart' button not found")
        except Exception as e:
            Utility.error_log(f"Failed to find 'Add to Cart' button: {e}")
            raise

    def draw_bounding_box(self, image_path: str, coordinates: Tuple[int, int, int, int],
                          output_path: str = None) -> np.ndarray:
        padding = 0
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to load image using OpenCV.")

            x, y, w, h = coordinates
            cv2.rectangle(img, (x, y), (x + w + padding, y + h + padding), (0, 255, 0), 2)

            if output_path:
                cv2.imwrite(output_path, img)
                Utility.log(f"Bounding box drawn and image saved to {output_path}.")
            else:
                return img
        except Exception as e:
            Utility.error_log(f"Failed to draw bounding box on image: {e}")
            raise

    def detect_add_to_cart_button(self, image_path: str, output_path: str = None) -> Optional[Rectangle]:
        try:
            image_content = Utility.read_image(image_path)
            texts = self.detect_text(image_content)
            x, y, w, h = self.find_add_to_cart(texts)
            _ = self.draw_bounding_box(image_path, (x, y, w, h), output_path)
            if output_path is None:
                Utility.error_log("Failed to find element location")
                return None
            else:
                Utility.log(f"Element found at location x={x}, y={y}, h={h}, w={w}")
                return Rectangle(x=x, y=y, w=w, h=h) #json.dumps({"x": x, "y": y, "w": w, "h": h})
        except Exception as e:
            Utility.error_log(f"Failed to detect 'Add to Cart' button: {e}")

    # Function to detect objects using Google Cloud Vision API
    def detect_text2(self, image_path):
        """Detects text in an image using Google Cloud Vision API."""
        client = self.__client
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        # Return the first result which should be the overall text detected
        if texts:
            return texts[0].description
        else:
            return None

    def detect_objects(self, image_path):
        """Detects objects in an image using Google Cloud Vision API."""
        client = self.__client
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        objects_response = client.object_localization(image=image)
        objects = objects_response.localized_object_annotations
        # Return objects found in the image
        return objects

    # Function to find related buttons based on detected text and objects
    def find_related_buttons(self, image_path):
        """Finds the location of buttons related to adding items to a cart on a webpage screenshot."""
        # Detect text in the image
        detected_text = self.detect_text2(image_path)
        # Check if detected text contains relevant keywords
        relevant_keywords = ['add', 'cart', 'buy', 'item', 'now']
        if detected_text and any(keyword in detected_text.lower() for keyword in relevant_keywords):
            # If text matches, return coordinates or perform further validation
            return {'text_detection': detected_text}
        # If no text match, detect objects (assuming the button is a recognized object)
        detected_objects = self.detect_objects(image_path)
        # Look for objects that might represent buttons related to adding items
        related_objects = []
        i=0
        for obj in detected_objects:
            object_name = obj.name.lower()
            for keyword in relevant_keywords:
                if keyword in object_name:
                    related_objects.append({
                        'name': obj.name,
                        'bounding_box': obj.bounding_poly.normalized_vertices
                    })
                    self.draw_bounding_box(image_path, obj.bounding_poly.normalized_vertices,image_path.replace('.png', keyword + i + '.png'))
                    i += 1
                    break
        if related_objects:
            # Return the list of related objects found
            return {'object_detection': related_objects}
        # If no relevant text or objects found, return None or handle as needed
        return None