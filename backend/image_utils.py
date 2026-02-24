"""Image utilities with EXIF orientation handling."""
import os
import sys
import cv2
import numpy as np

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def safe_imread(image_path):
    """
    cv2.imread replacement that handles Unicode/non-ASCII paths on Windows.
    OpenCV's imread uses the narrow Windows API and silently returns None for
    paths containing characters outside the system ANSI codepage (e.g. U+202F).
    Reading via Python's open() + cv2.imdecode avoids this limitation.
    """
    try:
        with open(image_path, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def safe_imwrite(image_path, img):
    """
    cv2.imwrite replacement that handles Unicode/non-ASCII paths on Windows.
    Returns True on success, False on failure.
    """
    try:
        ext = os.path.splitext(image_path)[1].lower() or '.png'
        success, buf = cv2.imencode(ext, img)
        if not success:
            return False
        with open(image_path, 'wb') as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def load_image(image_path):
    """Load image with EXIF orientation correction. Returns (image, width, height)."""
    if HAS_PIL:
        return _load_with_pil_exif(image_path)
    else:
        # Fallback to OpenCV without EXIF handling (use safe_imread for Unicode path support)
        img = safe_imread(image_path)
        if img is None:
            return None, 0, 0
        h, w = img.shape[:2]
        return img, w, h


def _load_with_pil_exif(image_path):
    """Load with PIL for EXIF handling, convert to OpenCV format."""
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
        try:
            sys.stderr.buffer.write(
                f"Warning: PIL load failed for {image_path}: {e}\n".encode("utf-8", errors="replace")
            )
        except Exception:
            pass
        # Fallback: use safe_imread which handles Unicode paths on Windows
        img = safe_imread(image_path)
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
