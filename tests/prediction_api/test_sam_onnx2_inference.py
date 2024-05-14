import json
import logging
import unittest
from unittest.mock import patch

import PIL.Image
import numpy as np

from samgis_core import MODEL_FOLDER
from samgis_core.prediction_api import sam_onnx_inference, sam_onnx2
from samgis_core.prediction_api.sam_onnx2 import SegmentAnythingONNX2
from samgis_core.utilities.constants import MODEL_ENCODER_NAME, MODEL_DECODER_NAME
from samgis_core.utilities.plot_images import helper_imshow_output_expected
from tests import TEST_EVENTS_FOLDER


instance_sam_onnx = SegmentAnythingONNX2(
    encoder_model_path=MODEL_FOLDER / MODEL_ENCODER_NAME,
    decoder_model_path=MODEL_FOLDER / MODEL_DECODER_NAME
)
np_img = np.load(TEST_EVENTS_FOLDER / "samexporter_predict" / "oceania" / "img.npy")
prompt = [{
    "type": "point",
    "data": [934, 510],
    "label": 0
}]


def count_sum_difference_less_than(a: np.ndarray, b: np.ndarray, **kwargs) -> tuple[float, float]:
    diff = np.sum(a == b, **kwargs)
    absolute_count = a.size - diff
    relative_count = (a.size - diff) / a.size
    return relative_count, absolute_count


def assert_sum_difference_less_than(a: np.ndarray, b: np.ndarray, rtol=1e-05, **kwargs):
    """
    assert than the relative (in percentual) count of different values between arrays a and b are less than given 'rtol'
    parameter. Right now don't evaluate absolute number of values
    """
    relative_count, absolute_count = count_sum_difference_less_than(a, b, **kwargs)
    try:
        assert relative_count < rtol
    except AssertionError as ae:
        logging.error(f"Mismatched elements: {absolute_count} / {a.size} ({relative_count:.02f})")
        logging.error(f"Max absolute difference: {rtol*a.size}")
        logging.error(f"Max relative difference: {rtol}")
        helper_imshow_output_expected(a, b, "a", "b")
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


class TestSamOnnx2Inference(unittest.TestCase):
    def test_get_raster_inference_with_embedding_from_dict_empty_dict_not_mocked_img_big(self):
        img = PIL.Image.open(TEST_EVENTS_FOLDER / "samexporter_predict" / "teglio" / "teglio_1280x960.jpg")
        img = img.convert("RGB")
        local_prompt = [{"type": "point", "data": [960, 840], "label": 0}]
        embedding_dict_test = {}
        name_key = "embedding_key_test_1280x960"
        output_mask, len_inference_out = sam_onnx_inference.get_raster_inference_with_embedding_from_dict(
            img=img,
            prompt=local_prompt,
            models_instance=instance_sam_onnx,
            model_name="mobile_sam",
            embedding_key=name_key,
            embedding_dict=embedding_dict_test
        )
        mask = np.load(TEST_EVENTS_FOLDER / "samexporter_predict" / "teglio" / "mask.npy")
        allclose_perc = 0.5
        assert_sum_difference_less_than(output_mask, mask, rtol=allclose_perc)
        assert len_inference_out == 1

        assert len(embedding_dict_test) == 1
        assert name_key in list(embedding_dict_test.keys())
        embedding_dict_test_value = embedding_dict_test[name_key]
        image_embedding = embedding_dict_test_value["image_embedding"]
        assert isinstance(image_embedding, np.ndarray)
        assert image_embedding.shape == (1, 256, 64, 64)
        assert isinstance(embedding_dict_test_value["original_size"], tuple)
        assert isinstance(embedding_dict_test_value["resized_size"], tuple)
        assert embedding_dict_test_value["resized_size"] == (1024, 768)

    def test_get_raster_inference_with_embedding_from_dict_empty_dict_not_mocked(self):
        name_fn = "samexporter_predict"

        with open(TEST_EVENTS_FOLDER / f"{name_fn}.json") as tst_json:
            inputs_outputs = json.load(tst_json)

            n_keys = 1
            allclose_perc = 0.5
            embedding_dict_test = {}

            for k, input_output in inputs_outputs.items():
                img = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "img.npy", allow_pickle=False)
                mask = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "mask.npy", allow_pickle=False)
                local_prompt = input_output["input"]["prompt"]
                model_name = input_output["input"]["model_name"]
                logging.info(f"img.shape: {img.shape}.")

                output_mask, len_inference_out = sam_onnx_inference.get_raster_inference_with_embedding_from_dict(
                    img=img,
                    prompt=local_prompt,
                    models_instance=instance_sam_onnx,
                    model_name=model_name,
                    embedding_key=f"embedding_key_test{n_keys}",
                    embedding_dict=embedding_dict_test
                )
                assert_sum_difference_less_than(output_mask, mask, rtol=allclose_perc)
                assert len_inference_out == input_output["output"]["n_predictions"]
                assert_helper_get_raster_inference_with_embedding_from_dict(
                    embedding_dict_test, n_keys, f"embedding_key_test{n_keys}"
                )
                n_keys += 1
                assert len(embedding_dict_test) == n_keys-1

    @patch.object(sam_onnx2, "SegmentAnythingONNX2")
    def test_get_raster_inference(self, segment_anything_onnx_mocked):
        name_fn = "samexporter_predict"

        with open(TEST_EVENTS_FOLDER / f"{name_fn}.json") as tst_json:
            inputs_outputs = json.load(tst_json)
            for k, input_output in inputs_outputs.items():
                model_mocked = segment_anything_onnx_mocked()

                img = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "img.npy")
                inference_out = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "inference_out.npy")
                mask = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "mask.npy")
                local_prompt = input_output["input"]["prompt"]
                model_name = input_output["input"]["model_name"]

                model_mocked.embed.return_value = np.array(img)
                model_mocked.embed.side_effect = None
                model_mocked.predict_masks.return_value = inference_out
                model_mocked.predict_masks.side_effect = None
                print(f"k:{k}.")
                output_mask, len_inference_out = sam_onnx_inference.get_raster_inference(
                    img=img,
                    prompt=local_prompt,
                    models_instance=model_mocked,
                    model_name=model_name
                )
                assert np.array_equal(output_mask, mask)
                assert len_inference_out == input_output["output"]["n_predictions"]
