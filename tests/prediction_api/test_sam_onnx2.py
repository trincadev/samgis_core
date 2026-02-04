import unittest

import numpy as np
from PIL import Image
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

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
# plot_images.imshow_raster(img_pil, "img_pil", show=True, debug=True)
image_embedding = instance_sam_onnx.encode(img_pil)
expected_hash_list = [
    b'LiWr6QRdwKWHONi37y+AgIM//SgaFvXgWlX844zckcU=',
    b'14pi7a6FGQgFN4Zne9uRXAg1vCt6QA/pqQrrLQ66weo=',
    b'5Y00HY9+gZe15U3XMLh+U/Zl5qa0tRuKHrzkZniZu7U='
]


def check_hash(actual_hash: str | bytes, expected_hash: bytes):
    if actual_hash != expected_hash:
        raise ValueError(f"wrong hash: '{actual_hash}' != '{expected_hash}'")
    


class TestSegmentAnythingONNX2(unittest.TestCase):
    def test_apply_coords(self):
        from samgis_core.prediction_api import sam_onnx2

        onnx_coords = np.array([[[321., 230.], [0., 0.]]])
        output_coords = sam_onnx2.apply_coords(onnx_coords, image_embedding)
        np.testing.assert_array_equal(
            actual=output_coords,
            desired=np.array([[[256.8, 184.], [0., 0.]]], dtype=np.float32),
            verbose=True,
            strict=True,
            err_msg="Arrays are not equal"
        )

    def test_preprocess_image_ndarray(self):
        resized_image = instance_sam_onnx.preprocess_image(np_img)
        hash_img = hash_calculate(np.array(resized_image), is_file=False)
        check_hash(hash_img, b'uP7LGlpKJ4xc+akIN+maJGHprdpVocpNtOVYCmAJzbw=')

    def test_preprocess_image_pil(self):
        input_pil_test = Image.fromarray(np_img)
        resized_image = instance_sam_onnx.preprocess_image(input_pil_test)
        hash_img = hash_calculate(np.array(resized_image), is_file=False)
        check_hash(hash_img, b'uP7LGlpKJ4xc+akIN+maJGHprdpVocpNtOVYCmAJzbw=')

    def test_encoder(self):
        img = image_embedding["image_embedding"]
        if image_embedding["original_size"] != (1280, 960):
            raise ValueError("wrong original_size!")
        if image_embedding["resized_size"] != (1024, 768):
            raise ValueError("wrong resized_size")
        hash_img = hash_calculate(np.array(img), is_file=False)
        self.assertIn(hash_img, expected_hash_list)


    def test_encode_predict_masks_ok(self):
        img_embedding = image_embedding["image_embedding"]
        test_logger.info(f"embedding type: {type(image_embedding)}.")
        hash_img = hash_calculate(img_embedding, is_file=False)
        self.assertIn(hash_img, expected_hash_list)
        output_inference = instance_sam_onnx.predict_masks(image_embedding, prompt)

        # here there is at least one output mask, created from the inference output
        output_mask_np = (output_inference[0][0] > 0).astype(np.uint8) * 255
        output_mask = Image.fromarray(output_mask_np)
        # output_mask.save(TEST_EVENTS_FOLDER / "samexporter_predict" / "teglio" / "teglio_1280x960_mask2.png")
        expected_mask = np.array(mask_pil)
        hash_expected_mask = hash_calculate(expected_mask, is_file=False)
        check_hash(hash_expected_mask, b'NDp9r4fI99jqt3aQnkeez8b0/w24tdGIWXKVz6qRWUU=')
        all_close_perc = 0.85
        try:
            helper_assertions.assert_sum_difference_less_than(output_mask_np, expected_mask, rtol=all_close_perc)
        except AssertionError as ae:
            from samgis_core.utilities import plot_images
            plot_images.helper_imshow_output_expected(
                [output_mask_np, expected_mask],
                ["output_mask", "expected"],
                show=True, debug=True)
            raise ae

    def test_encode_predict_masks_ex1(self):
        with self.assertRaises(ValueError):
            try:
                np_input = np.zeros((10, 10))
                instance_sam_onnx.encode(np_input)
            except ValueError as ve:
                test_logger.error(f"ValueError: '{ve}'")
                msg = "operands could not be broadcast together with shapes (1024,1024) (3,) "
                if str(object=ve) != msg:
                    return "wrong exception message, DON'T raise the ValueError() exception to let it fail the test!"
                raise ve

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
                if str(ie) != "list index out of range":
                    raise ValueError("wrong exception message!")
                raise ie
