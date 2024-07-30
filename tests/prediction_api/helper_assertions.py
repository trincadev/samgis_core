import numpy as np

from samgis_core.utilities.plot_images import helper_imshow_output_expected
from tests import test_logger


def count_sum_difference_less_than(a: np.ndarray, b: np.ndarray, **kwargs) -> tuple[float, float]:
    # check that input a, b arguments are numpy arrays because have 'shape' property
    try:
        a_shape, b_shape = a.shape, b.shape
    except AttributeError as attex:
        raise attex
    diff = np.sum(a == b, **kwargs)
    absolute_count = a.size - diff
    relative_count = 100 * (a.size - diff) / a.size
    return relative_count, absolute_count


def assert_sum_difference_less_than(a: np.ndarray, b: np.ndarray, rtol=1e-05, **kwargs):
    """
    assert than the relative (in percentual) count of different values between arrays a and b are less than given 'rtol'
    parameter. Right now don't evaluate absolute number of values
    """
    relative_count, absolute_count = count_sum_difference_less_than(a, b, **kwargs)
    test_logger.info(f"array a, shape:{a.shape}.")
    test_logger.info(f"array b, shape:{b.shape}.")
    try:
        print(f"absolute_value:{absolute_count}, relative_count:{relative_count}, rtol:{rtol}.")
        test_logger.info(f"absolute_value:{absolute_count}, relative_count:{relative_count}, rtol:{rtol}.")
        assert relative_count < rtol
    except AssertionError as ae:
        msg = f"""Mismatched elements: {absolute_count} / {a.size} ({relative_count:.02f}),
        Max absolute difference: {rtol*a.size},
        Max relative difference: {rtol}
        """
        test_logger.error(f"ae:{ae}.")
        test_logger.error(msg)
        # here it seems not working, include this function in a try/except and
        # use helper_imshow_output_expected() before raising the exception
        helper_imshow_output_expected([a, b], ["a", "b"], show=True)  # , debug=True)
        raise ae


def assert_helper_get_raster_inference_with_embedding_from_dict(
        embedding_dict_test: dict,
        n_keys: int,
        name_key,
        shape_image_embedding=(1, 256, 64, 64),
        expected_resized_size=(1024, 684)
    ):
    assert len(embedding_dict_test) == n_keys
    assert name_key in list(embedding_dict_test.keys())
    embedding_dict_test_value = embedding_dict_test[name_key]
    image_embedding = embedding_dict_test_value["image_embedding"]
    assert isinstance(image_embedding, np.ndarray)
    assert image_embedding.shape == shape_image_embedding
    assert isinstance(embedding_dict_test_value["original_size"], tuple)
    assert isinstance(embedding_dict_test_value["resized_size"], tuple)
    assert embedding_dict_test_value["resized_size"] == expected_resized_size
