import json
import unittest
from unittest.mock import patch

import PIL.Image
import numpy as np

from samgis_core import MODEL_FOLDER
from samgis_core.prediction_api import sam_onnx_inference, sam_onnx2
from samgis_core.prediction_api.sam_onnx2 import SegmentAnythingONNX2
from samgis_core.utilities.constants import MODEL_ENCODER_NAME, MODEL_DECODER_NAME
from . import helper_assertions
from tests import TEST_EVENTS_FOLDER, test_logger


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
        expected_mask = PIL.Image.open(TEST_EVENTS_FOLDER / "samexporter_predict" / "teglio" / "teglio_1280x960_mask2.png")
        expected_mask = np.array(expected_mask)
        allclose_perc = 0.05  # percentage
        helper_assertions.assert_sum_difference_less_than(output_mask, expected_mask, rtol=allclose_perc)
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
            allclose_perc = 0.05
            embedding_dict_test = {}

            for k, input_output in inputs_outputs.items():
                img = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "img.npy", allow_pickle=False)
                mask = np.load(TEST_EVENTS_FOLDER / f"{name_fn}" / k / "mask.npy", allow_pickle=False)
                local_prompt = input_output["input"]["prompt"]
                model_name = input_output["input"]["model_name"]
                test_logger.info(f"img.shape: {img.shape}.")

                output_mask, len_inference_out = sam_onnx_inference.get_raster_inference_with_embedding_from_dict(
                    img=img,
                    prompt=local_prompt,
                    models_instance=instance_sam_onnx,
                    model_name=model_name,
                    embedding_key=f"embedding_key_test{n_keys}",
                    embedding_dict=embedding_dict_test
                )
                helper_assertions.assert_sum_difference_less_than(output_mask, mask, rtol=allclose_perc)
                assert len_inference_out == input_output["output"]["n_predictions"]
                helper_assertions.assert_helper_get_raster_inference_with_embedding_from_dict(
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
                test_logger.info(f"k:{k}.")
                output_mask, len_inference_out = sam_onnx_inference.get_raster_inference(
                    img=img,
                    prompt=local_prompt,
                    models_instance=model_mocked,
                    model_name=model_name
                )
                try:
                    assert np.array_equal(output_mask, mask)
                except Exception as ex:
                    print(f"k:{k}, ex::{ex}.")
                    test_logger.error(f"k:{k}, ex::{ex}.")
                    allclose_perc = 0.002  # percentage
                    helper_assertions.assert_sum_difference_less_than(output_mask, mask, rtol=allclose_perc)
                assert len_inference_out == input_output["output"]["n_predictions"]
