"""
Image utilities for consistent image loading across training and inference.
Handles EXIF orientation to match browser display.
"""
import cv2
import numpy as np

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_image(image_path):
    """
    Load an image with EXIF orientation correction.

    Browsers auto-rotate images based on EXIF orientation metadata,
    but OpenCV doesn't. This function ensures the image is loaded
    in the same orientation that browsers display.

    Returns: (image, width, height) or (None, 0, 0) if load fails
    """
    if HAS_PIL:
        return _load_with_pil_exif(image_path)
    else:
        # Fallback to OpenCV without EXIF handling
        img = cv2.imread(image_path)
        if img is None:
            return None, 0, 0
        h, w = img.shape[:2]
        return img, w, h


def _load_with_pil_exif(image_path):
    """Load image using PIL to handle EXIF orientation, then convert to OpenCV format."""
    try:
        pil_img = Image.open(image_path)

        # Check for EXIF orientation
        try:
            exif = pil_img._getexif()
            if exif:
                # Find the orientation tag
                orientation_key = None
                for tag_id, tag_name in TAGS.items():
                    if tag_name == 'Orientation':
                        orientation_key = tag_id
                        break

                if orientation_key and orientation_key in exif:
                    orientation = exif[orientation_key]

                    # Apply rotation/flip based on EXIF orientation
                    if orientation == 2:
                        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif orientation == 4:
                        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
                    elif orientation == 5:
                        pil_img = pil_img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 6:
                        pil_img = pil_img.rotate(-90, expand=True)
                    elif orientation == 7:
                        pil_img = pil_img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 8:
                        pil_img = pil_img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # No EXIF data or orientation tag
            pass

        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Convert PIL image to OpenCV format (BGR)
        img_array = np.array(pil_img)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]
        return img, w, h

    except Exception as e:
        print(f"Warning: PIL load failed for {image_path}: {e}")
        # Fallback to OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return None, 0, 0
        h, w = img.shape[:2]
        return img, w, h


def get_image_dimensions(image_path):
    """Get image dimensions after EXIF correction."""
    img, w, h = load_image(image_path)
    return w, h


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python image_utils.py <image_path>")
        sys.exit(1)

    img, w, h = load_image(sys.argv[1])
    if img is not None:
        print(json.dumps({"width": w, "height": h, "has_pil": HAS_PIL}))
    else:
        print(json.dumps({"error": "Failed to load image"}))
