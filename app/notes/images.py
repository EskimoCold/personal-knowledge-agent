from io import BytesIO

from PIL import Image, ImageOps

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
IMAGE_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png"}


def decode_photo(data: bytes) -> tuple[Image.Image, str]:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Upload a photo up to 8 MiB.")
    try:
        with Image.open(BytesIO(data), formats=tuple(IMAGE_TYPES)) as raw:
            if raw.width * raw.height > MAX_IMAGE_PIXELS:
                raise ValueError("Upload a photo up to 16 megapixels.")
            if getattr(raw, "is_animated", False):
                raise ValueError("Upload a still JPEG or PNG photo.")
            raw.verify()
        with Image.open(BytesIO(data), formats=tuple(IMAGE_TYPES)) as raw:
            return ImageOps.exif_transpose(raw).convert("RGB"), IMAGE_TYPES[raw.format]
    except (Image.DecompressionBombError, OSError, SyntaxError):
        raise ValueError("Upload a valid JPEG or PNG photo.") from None
