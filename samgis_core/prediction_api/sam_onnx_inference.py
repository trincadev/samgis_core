from numpy import array as np_array, uint8, zeros, ndarray

from samgis_core import app_logger, MODEL_FOLDER
from samgis_core.prediction_api.sam_onnx2 import SegmentAnythingONNX2
from samgis_core.utilities.constants import MODEL_ENCODER_NAME, MODEL_DECODER_NAME
from samgis_core.utilities.type_hints import ListDict, PIL_Image, TupleNdarrayInt, EmbeddingPILDict


def get_raster_inference(
        img: PIL_Image | ndarray, prompt: ListDict, models_instance: SegmentAnythingONNX2, model_name: str
) -> TupleNdarrayInt:
    """
    Get inference output for a given image using a SegmentAnythingONNX model

    Args:
        img: input PIL Image
        prompt: list of prompt dict
        models_instance: SegmentAnythingONNX instance model
        model_name: model name string

    Returns:
        raster prediction mask, prediction number
    """
    np_img = np_array(img)
    app_logger.info(f"img type {type(np_img)}, prompt:{prompt}.")
    app_logger.debug(f"onnxruntime input shape/size (shape if PIL) {np_img.size}.")
    try:
        app_logger.debug(f"onnxruntime input shape (NUMPY) {np_img.shape}.")
    except Exception as e_shape:
        app_logger.error(f"e_shape:{e_shape}.")
    app_logger.info(f"instantiated model {model_name}, ENCODER {MODEL_ENCODER_NAME}, "
                    f"DECODER {MODEL_DECODER_NAME} from {MODEL_FOLDER}: Creating embedding...")
    embedding = models_instance.encode(np_img)
    app_logger.debug(f"embedding created, running predict_masks with prompt {prompt}...")
    return get_raster_inference_using_existing_embedding(embedding, prompt, models_instance)


def get_inference_embedding(
        img: PIL_Image | ndarray, models_instance: SegmentAnythingONNX2, model_name: str, embedding_key: str,
        embedding_dict: EmbeddingPILDict) -> EmbeddingPILDict:
    """add an embedding to the embedding dict if needed

    Args:
        img: input PIL Image
        models_instance: SegmentAnythingONNX instance model
        model_name: model name string
        embedding_key: embedding id
        embedding_dict: embedding dict object

    Returns:
        raster dict
    """
    if embedding_key in embedding_dict:
        app_logger.info("found embedding in dict...")
    if embedding_key not in embedding_dict:
        np_img = np_array(img)
        app_logger.info(f"prepare embedding using img type {type(np_img)}.")
        app_logger.debug(f"onnxruntime input shape/size (shape if PIL) {np_img.size}.")
        try:
            app_logger.debug(f"onnxruntime input shape (NUMPY) {np_img.shape}.")
        except Exception as e_shape:
            app_logger.error(f"e_shape:{e_shape}.")
        app_logger.info(f"instantiated model {model_name}, ENCODER {MODEL_ENCODER_NAME}, "
                        f"DECODER {MODEL_DECODER_NAME} from {MODEL_FOLDER}: Creating embedding...")
        embedding = models_instance.encode(np_img)
        embedding_dict[embedding_key] = embedding
    return embedding_dict


def get_raster_inference_using_existing_embedding(
        embedding: dict, prompt: ListDict, models_instance: SegmentAnythingONNX2) -> TupleNdarrayInt:
    """
    Get inference output for a given image using a SegmentAnythingONNX model, using an existing embedding instead of a
    new ndarray or PIL image

    Args:
        embedding: dict
        prompt: list of prompt dict
        models_instance: SegmentAnythingONNX instance model

    Returns:
        raster prediction mask, prediction number
    """
    app_logger.info(f"using existing embedding of type {type(embedding)}.")
    inference_out = models_instance.predict_masks(embedding, prompt)
    len_inference_out = len(inference_out[0, :, :, :])
    app_logger.info(f"Created {len_inference_out} prediction_masks,"
                    f"shape:{inference_out.shape}, dtype:{inference_out.dtype}.")
    mask = zeros((inference_out.shape[2], inference_out.shape[3]), dtype=uint8)
    for n, m in enumerate(inference_out[0, :, :, :]):
        app_logger.debug(f"{n}th of prediction_masks shape {inference_out.shape}"
                         f" => mask shape:{mask.shape}, {mask.dtype}.")
        mask[m > 0.0] = 255
    return mask, len_inference_out


def get_raster_inference_with_embedding_from_dict(
        img: PIL_Image | ndarray, prompt: ListDict, models_instance: SegmentAnythingONNX2, model_name: str,
        embedding_key: str, embedding_dict: dict) -> TupleNdarrayInt:
    """
    Get inference output using a SegmentAnythingONNX model, but get the image embedding from the given embedding dict
     instead of creating a new embedding. This function needs the img argument to update the embedding dict if necessary

    Args:
        img: input PIL Image
        prompt: list of prompt dict
        models_instance: SegmentAnythingONNX instance model
        model_name: model name string
        embedding_key: embedding id
        embedding_dict: embedding images dict

    Returns:
        raster prediction mask, prediction number
    """
    app_logger.info(f"handling embedding using key {embedding_key}.")
    embedding_dict = get_inference_embedding(img, models_instance, model_name, embedding_key, embedding_dict)
    app_logger.info(f"getting embedding with key {embedding_key} from dict...")
    embedding = embedding_dict[embedding_key]
    n_keys = len(embedding_dict)
    app_logger.info(f"embedding created ({n_keys} keys in embedding dict), running predict_masks with prompt {prompt}.")
    return get_raster_inference_using_existing_embedding(embedding, prompt, models_instance)
