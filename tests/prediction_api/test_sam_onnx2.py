import unittest

import numpy as np
from PIL import Image

from samgis_core import MODEL_FOLDER
from samgis_core.prediction_api.sam_onnx2 import SegmentAnythingONNX2
from samgis_core.utilities.constants import MODEL_ENCODER_NAME, MODEL_DECODER_NAME
from samgis_core.utilities.utilities import hash_calculate
from tests import TEST_EVENTS_FOLDER, test_logger
from tests.prediction_api import helper_assertions


instance_sam_onnx = SegmentAnythingONNX2(
    encoder_model_path=MODEL_FOLDER / MODEL_ENCODER_NAME,
    decoder_model_path=MODEL_FOLDER / MODEL_DECODER_NAME
)
np_img = np.load(TEST_EVENTS_FOLDER / "samexporter_predict" / "oceania" / "img.npy")
prompt = [{
    "type": "point",
    "data": [321, 230],
    "label": 0
}]
img_pil = Image.open(TEST_EVENTS_FOLDER / "samexporter_predict" / "teglio" / "teglio_1280x960.jpg")
mask_pil = Image.open(TEST_EVENTS_FOLDER / "samexporter_predict" / "teglio" / "teglio_1280x960_mask.png")
img_pil = img_pil.convert("RGB")
image_embedding = instance_sam_onnx.encode(img_pil)


class TestSegmentAnythingONNX2(unittest.TestCase):
    def test_apply_coords(self):
        from samgis_core.prediction_api import sam_onnx2

        onnx_coords = np.array([[[321., 230.], [0., 0.]]])
        output_coords = sam_onnx2.apply_coords(onnx_coords, image_embedding)
        np.testing.assert_array_equal(
            x=output_coords,
            y=np.array([[[256.8, 184.], [0., 0.]]], dtype=np.float32),
            verbose=True,
            strict=True,
            err_msg="Arrays are not equal"
        )

    def test_preprocess_image_ndarray(self):
        resized_image = instance_sam_onnx.preprocess_image(np_img)
        assert hash_calculate(np.array(resized_image)) == b'uP7LGlpKJ4xc+akIN+maJGHprdpVocpNtOVYCmAJzbw='

    def test_preprocess_image_pil(self):
        input_pil_test = Image.fromarray(np_img)
        resized_image = instance_sam_onnx.preprocess_image(input_pil_test)
        assert hash_calculate(np.array(resized_image)) == b'uP7LGlpKJ4xc+akIN+maJGHprdpVocpNtOVYCmAJzbw='

    def test_encoder(self):
        img = image_embedding["image_embedding"]
        assert image_embedding["original_size"] == (1280, 960)
        assert image_embedding["resized_size"] == (1024, 768)
        assert hash_calculate(np.array(img)) == b'14pi7a6FGQgFN4Zne9uRXAg1vCt6QA/pqQrrLQ66weo='

    def  test_encode_predict_masks_ok(self):
        img_embedding = image_embedding["image_embedding"]
        test_logger.info(f"embedding type: {type(image_embedding)}.")
        assert hash_calculate(img_embedding) == b'14pi7a6FGQgFN4Zne9uRXAg1vCt6QA/pqQrrLQ66weo='
        output_inference = instance_sam_onnx.predict_masks(image_embedding, prompt)

        # here there is at least one output mask, created from the inference output
        output_mask_np = (output_inference[0][0] > 0).astype(np.uint8) * 255
        output_mask = Image.fromarray(output_mask_np)
        # output_mask.save(TEST_EVENTS_FOLDER / "samexporter_predict" / "teglio" / "teglio_1280x960_mask2.png")
        expected_mask = np.array(mask_pil)
        hash_expected_mask = hash_calculate(expected_mask)
        assert hash_expected_mask == b'yfg1ZenWoZt+f4Qgv4YQ21/VR+T6VfgkZyXm6xhdT+w='
        allclose_perc = 0.002
        helper_assertions.assert_sum_difference_less_than(output_mask_np, expected_mask, rtol=allclose_perc)

    def test_encode_predict_masks_ex1(self):
        with self.assertRaises(Exception):
            try:
                np_input = np.zeros((10, 10))
                instance_sam_onnx.encode(np_input)
            except Exception as e:
                test_logger.error(f"e:{e}.")
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
        d = {"a": 1}
        for x in d:
            test_logger.info(x)

        with self.assertRaises(IndexError):
            try:
                instance_sam_onnx.predict_masks(embedding, wrong_prompt)
            except IndexError as ie:
                test_logger.error(ie)
                assert str(ie) == "list index out of range"
                raise ie
