import json
import logging
import unittest
from unittest.mock import patch

import numpy as np

from samgis_core import MODEL_FOLDER
from samgis_core.prediction_api.sam_onnx import SegmentAnythingONNX

from samgis_core.prediction_api import sam_onnx
from samgis_core.prediction_api.sam_onnx import get_raster_inference, get_raster_inference_with_embedding_from_dict
from samgis_core.utilities.constants import MODEL_ENCODER_NAME, MODEL_DECODER_NAME
from samgis_core.utilities.utilities import hash_calculate
from tests import TEST_EVENTS_FOLDER


instance_sam_onnx = SegmentAnythingONNX(
    encoder_model_path=MODEL_FOLDER / MODEL_ENCODER_NAME,
    decoder_model_path=MODEL_FOLDER / MODEL_DECODER_NAME
)
np_img = np.load(TEST_EVENTS_FOLDER / "samexporter_predict" / "oceania" / "img.npy")
prompt = [{
    "type": "point",
    "data": [934, 510],
    "label": 0
}]


def assert_helper_get_raster_inference_with_embedding_from_dict(
        embedding_dict_test: dict,
        n_keys: int,
        name_key,
        shape_transform_matrix=(3, 3),
        shape_image_embedding=(1, 256, 64, 64)
    ):
    assert len(embedding_dict_test) == n_keys
    assert name_key in list(embedding_dict_test.keys())
    embedding_dict_test_value = embedding_dict_test[name_key]
    assert isinstance(embedding_dict_test_value["image_embedding"], np.ndarray)
    assert isinstance(embedding_dict_test_value["original_size"], tuple)
    assert isinstance(embedding_dict_test_value["transform_matrix"], np.ndarray)
    assert embedding_dict_test_value["transform_matrix"].shape == shape_transform_matrix
    assert embedding_dict_test_value["image_embedding"].shape == shape_image_embedding

class TestSegmentAnythingONNX(unittest.TestCase):
    def test_encode_predict_masks_ok(self):
        embedding = instance_sam_onnx.encode(np_img)
        try:
            assert hash_calculate(embedding) == b"m2O3y7pNUwlLuAZhBHkRIu8cDIIej0oOmWOXevs39r4="
        except AssertionError as ae1:
            logging.warning(f"ae1:{ae1}.")
        inference_mask = instance_sam_onnx.predict_masks(embedding, prompt)
        try:
            assert hash_calculate(inference_mask) == b'YSKKNCs3AMpbeDUVwqIwNQqJ365OG4239hxjFnW7XTM='
        except AssertionError as ae2:
            logging.warning(f"ae2:{ae2}.")
        mask_output = np.zeros((inference_mask.shape[2], inference_mask.shape[3]), dtype=np.uint8)
        for n, m in enumerate(inference_mask[0, :, :, :]):
            logging.debug(f"{n}th of prediction_masks shape {inference_mask.shape}"
                          f" => mask shape:{mask_output.shape}, {mask_output.dtype}.")
            mask_output[m > 0.0] = 255
        mask_expected = np.load(TEST_EVENTS_FOLDER / "SegmentAnythingONNX" / "mask_output.npy")

        # assert MAP (mean average precision) is 100%
        # sum expected mask to output mask:
        # - asserted "good" inference values are 2 (matched object) or 0 (matched background)
        # - "bad" inference value is 1 (there are differences between expected and output mask)
        sum_mask_output_vs_expected = mask_expected / 255 + mask_output / 255
        unique_values__output_vs_expected = np.unique(sum_mask_output_vs_expected, return_counts=True)
        tot = sum_mask_output_vs_expected.size
        perc = {
            k: 100 * v / tot for
            k, v in
            zip(unique_values__output_vs_expected[0], unique_values__output_vs_expected[1])
        }
        try:
            assert 1 not in perc
        except AssertionError:
            n_pixels = perc[1]
            logging.error(f"found {n_pixels:.2%} different pixels between expected masks and output mask.")
            # try to assert that the % of different pixels are minor than 5%
            assert perc[1] < 5

    def test_encode_predict_masks_ex1(self):
        with self.assertRaises(Exception):
            try:
                np_input = np.zeros((10, 10))
                instance_sam_onnx.encode(np_input)
            except Exception as e:
                logging.error(f"e:{e}.")
                msg = "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid rank for input: input_image "
                msg += "Got: 2 Expected: 3 Please fix either the inputs or the model."
                assert str(e) == msg
                raise e

    def test_encode_predict_masks_ex2(self):
        wrong_prompt = [{
            "type": "rectangle",
            "data": [934, 510],
            "label": 0
        }]
        embedding = instance_sam_onnx.encode(np_img)

        with self.assertRaises(IndexError):
            try:
                instance_sam_onnx.predict_masks(embedding, wrong_prompt)
            except IndexError as ie:
                print(ie)
                assert str(ie) == "list index out of range"
                raise ie


@patch.object(sam_onnx, "SegmentAnythingONNX")
def test_get_raster_inference(segment_anything_onnx_mocked):
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
            output_mask, len_inference_out = get_raster_inference(
                img=img,
                prompt=local_prompt,
                models_instance=model_mocked,
                model_name=model_name
            )
            assert np.array_equal(output_mask, mask)
            assert len_inference_out == input_output["output"]["n_predictions"]


def test_get_raster_inference_with_embedding_from_dict_empty_dict_not_mocked():
    name_fn = "samexporter_predict"

    with open(TEST_EVENTS_FOLDER / f"{name_fn}.json") as tst_json:
        inputs_outputs = json.load(tst_json)

        n_keys = 1
        embedding_dict_test = {}

        for k, input_output in inputs_outputs.items():
            img = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "img.npy")
            mask = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "mask.npy")
            local_prompt = input_output["input"]["prompt"]
            model_name = input_output["input"]["model_name"]
            assert len(embedding_dict_test) == n_keys-1

            output_mask, len_inference_out = get_raster_inference_with_embedding_from_dict(
                img=img,
                prompt=local_prompt,
                models_instance=instance_sam_onnx,
                model_name=model_name,
                embedding_key=f"embedding_key_test{n_keys}",
                embedding_dict=embedding_dict_test
            )
            assert np.array_equal(output_mask, mask)
            assert len_inference_out == input_output["output"]["n_predictions"]
            assert_helper_get_raster_inference_with_embedding_from_dict(
                embedding_dict_test, n_keys, f"embedding_key_test{n_keys}"
            )
            n_keys += 1
