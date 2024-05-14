"""Various utilities (logger, time benchmark, args dump, numerical and stats info)"""
from copy import deepcopy

from numpy import ndarray, float32

from samgis_core import app_logger
from samgis_core.utilities.serialize import serialize
from samgis_core.utilities.type_hints import EmbeddingPILImage, PIL_Image


def _prepare_base64_input(sb):
    if isinstance(sb, str):
        # If there's any unicode here, an exception will be thrown and the function will return false
        return bytes(sb, 'ascii')
    elif isinstance(sb, bytes):
        return sb
    raise ValueError("Argument must be string or bytes")


def _is_base64(sb: str | bytes):
    import base64

    try:
        sb_bytes = _prepare_base64_input(sb)
        decoded = base64.b64decode(sb_bytes, validate=True)
        return base64.b64encode(decoded).decode("utf-8") == sb_bytes.decode("utf-8")
    except ValueError:
        return False


def base64_decode(s):
    """
    Decode base64 strings

    Args:
        s: input string

    Returns:
        decoded string
    """
    import base64

    if isinstance(s, str) and _is_base64(s):
        return base64.b64decode(s, validate=True).decode("utf-8")

    return s


def base64_encode(sb: str | bytes) -> bytes:
    """
    Encode input strings or bytes as base64

    Args:
        sb: input string or bytes

    Returns:
        base64 encoded bytes
    """
    import base64

    sb_bytes = _prepare_base64_input(sb)
    return base64.b64encode(sb_bytes)


def hash_calculate(arr) -> str | bytes:
    """
    Return computed hash from input variable (typically a numpy array).

    Args:
        arr: input variable

    Returns:
        computed hash from input variable
    """
    from hashlib import sha256
    from base64 import b64encode
    from numpy import ndarray as np_ndarray

    if isinstance(arr, np_ndarray):
        hash_fn = sha256(arr.data)
    elif isinstance(arr, dict):
        import json

        serialized = serialize(arr)
        variable_to_hash = json.dumps(serialized, sort_keys=True).encode('utf-8')
        hash_fn = sha256(variable_to_hash)
    elif isinstance(arr, str):
        try:
            hash_fn = sha256(arr)
        except TypeError:
            app_logger.warning(f"TypeError, re-try encoding arg:{arr},type:{type(arr)}.")
            hash_fn = sha256(arr.encode('utf-8'))
    elif isinstance(arr, bytes):
        hash_fn = sha256(arr)
    else:
        raise ValueError(f"variable 'arr':{arr} of type '{type(arr)}' not yet handled.")
    return b64encode(hash_fn.digest())


def convert_ndarray_to_pil(pil_image: PIL_Image | ndarray):
    from PIL import Image

    if isinstance(pil_image, ndarray):
        pil_image = Image.fromarray(pil_image)
    return pil_image


def apply_coords(coords: ndarray, embedding: EmbeddingPILImage):
    """
    Expects a numpy np_array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    orig_width, orig_height = embedding["original_size"]
    resized_width, resized_height = embedding["resized_size"]
    coords = deepcopy(coords).astype(float)

    coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
    coords[..., 1] = coords[..., 1] * (resized_height / orig_height)

    return coords.astype(float32)
