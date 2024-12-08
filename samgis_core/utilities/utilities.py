"""Various utilities (logger, time benchmark, args dump, numerical and stats info)"""
from copy import deepcopy

import numpy as np
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


def hash_calculate(arr_or_path, is_file: bool, read_mode: str = "rb") -> str | bytes:
    """
    Return computed hash from input variable (typically a numpy array).

    Args:
        arr_or_path: input variable
        is_file: whether input is a file or not
        read_mode: mode for reading file

    Returns:
        computed hash from input variable
    """
    from hashlib import sha256
    from base64 import b64encode
    from numpy import ndarray as np_ndarray

    if is_file:
        with open(arr_or_path, read_mode) as file_to_check:
            # read contents of the file
            arr_or_path = file_to_check.read()
            # # pipe contents of the file through
            # try:
            #     return hashlib.sha256(data).hexdigest()
            # except TypeError:
            #     app_logger.warning(
            #         f"TypeError, re-try encoding arg:{arr_or_path},type:{type(arr_or_path)}."
            #     )
            #     return hashlib.sha256(data.encode("utf-8")).hexdigest()

    if isinstance(arr_or_path, np_ndarray):
        hash_fn = sha256(arr_or_path.data)
    elif isinstance(arr_or_path, dict):
        import json

        serialized = serialize(arr_or_path)
        variable_to_hash = json.dumps(serialized, sort_keys=True).encode("utf-8")
        hash_fn = sha256(variable_to_hash)
    elif isinstance(arr_or_path, str):
        try:
            hash_fn = sha256(arr_or_path)
        except TypeError:
            app_logger.warning(
                f"TypeError, re-try encoding arg:{arr_or_path},type:{type(arr_or_path)}."
            )
            hash_fn = sha256(arr_or_path.encode("utf-8"))
    elif isinstance(arr_or_path, bytes):
        hash_fn = sha256(arr_or_path)
    else:
        raise ValueError(
            f"variable 'arr':{arr_or_path} of type '{type(arr_or_path)}' not yet handled."
        )
    return b64encode(hash_fn.digest())


def convert_ndarray_to_pil(pil_image: PIL_Image | ndarray):
    """
    Check if an image is a ndarray and then convert to a PIL Image instance.

    Args:
        pil_image: PIL image or ndarray

    Returns:
        PIL Image

    """
    from PIL import Image

    if isinstance(pil_image, ndarray):
        pil_image = Image.fromarray(pil_image)
    return pil_image


def apply_coords(coords: ndarray, embedding: EmbeddingPILImage):
    """
    Expects a numpy np_array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.

        Args:
            coords: coordinates ndarray
            embedding: PIL image embedding dict

        Returns:
            coordinates ndarray

    """
    orig_width, orig_height = embedding["original_size"]
    resized_width, resized_height = embedding["resized_size"]
    coords = deepcopy(coords).astype(float)

    coords[..., 0] = coords[..., 0] * (resized_width / orig_width)
    coords[..., 1] = coords[..., 1] * (resized_height / orig_height)

    return coords.astype(float32)


def normalize_array(arr: ndarray, new_h: int | float = 255., type_normalization: str = "int") -> ndarray:
    """
    Normalize numpy array between 0 and 'new_h' value. Default dtype of output array is int


    Args:
        arr: input numpy array
        new_h: max value of output array
        type_normalization: default dtype of output array

    Returns:
        numpy array
    """
    arr = arr.astype(float)
    arr_max = np.nanmax(arr)
    arr_min = np.nanmin(arr)
    scaled_arr = (arr - arr_min) / (arr_max - arr_min)
    multiplied_arr = scaled_arr * new_h
    return multiplied_arr.astype(int) if type_normalization == "int" else multiplied_arr
