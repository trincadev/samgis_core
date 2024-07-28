"""
Define a machine learning model executed by ONNX Runtime (https://onnxruntime.ai/)
for Segment Anything (https://segment-anything.com).
Modified from
- https://github.com/vietanhdev/samexporter/
- https://github.com/AndreyGermanov/sam_onnx_full_export/

Copyright (c) 2023 Viet Anh Nguyen, Andrey Germanov
Copyright (c) 2024-today Alessandro Trinca Tornidor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from numpy import array as np_array, pad as np_pad, zeros, ndarray, concatenate, float32
from onnxruntime import get_available_providers, InferenceSession

from samgis_core import app_logger
from samgis_core.utilities.constants import DEFAULT_INPUT_SHAPE
from samgis_core.utilities.type_hints import ListDict, EmbeddingPILImage, PIL_Image
from samgis_core.utilities.utilities import convert_ndarray_to_pil, apply_coords


class SegmentAnythingONNX2:
    """
    Segmentation model using Segment Anything.
    Compatible with onnxruntime 1.17.x and later
    """

    def __init__(self, encoder_model_path: str, decoder_model_path: str) -> None:
        self.target_size = DEFAULT_INPUT_SHAPE[1]
        self.input_size = DEFAULT_INPUT_SHAPE

        # Load models
        providers = get_available_providers()

        # Pop TensorRT Runtime due to crashing issues
        # TODO: Add back when TensorRT backend is stable
        providers = [p for p in providers if p != "TensorrtExecutionProvider"]

        if providers:
            app_logger.info(
                "Available providers for ONNXRuntime: %s", ", ".join(providers)
            )
        else:
            app_logger.warning("No available providers for ONNXRuntime")
        self.encoder_session = InferenceSession(
            encoder_model_path, providers=providers
        )
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        app_logger.info(f"encoder_input_name:{self.encoder_input_name}.")
        self.decoder_session = InferenceSession(
            decoder_model_path, providers=providers
        )

    @staticmethod
    def get_input_points(prompt: ListDict) -> tuple[ndarray]:
        """
        Get input points from a prompt dict list.

        Args:
            prompt: dict list

        Returns:
            tuple of points, labels ndarray ready for Segment Anything inference

        """
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append(
                    [mark["data"][2], mark["data"][3]]
                )  # bottom right
                labels.append(2)
                labels.append(3)
        points, labels = np_array(points), np_array(labels)
        return points, labels

    def encode(self, img: PIL_Image | ndarray) -> EmbeddingPILImage:
        """
        Calculate embedding and metadata for a single image.

        Args:
            img: input image to embed

        Returns:
            embedding image dict useful to store and cache image embeddings

        """
        resized_image = self.preprocess_image(img)
        padded_input_tensor = self.padding_tensor(resized_image)

        # 2. GET IMAGE EMBEDDINGS USING IMAGE ENCODER (`size` argument here is like ndarray `shape`)
        outputs = self.encoder_session.run(None, {"images": padded_input_tensor})
        image_embedding = outputs[0]
        img = convert_ndarray_to_pil(img)
        return {
            "image_embedding": image_embedding,
            "original_size": img.size,
            "resized_size": resized_image.size
        }

    def predict_masks(self, embedding: EmbeddingPILImage, prompt: ListDict) -> ndarray:
        """
        Predict masks for a single image.

        Args:
            embedding: input image embedding dict
            prompt: Segment Anything input prompt

        Returns:
            prediction masks ndarray; this should have (1, 1, **image.shape) shape

        """
        input_points, input_labels = self.get_input_points(prompt)

        # Add a batch index, concatenate a padding point, and transform.
        onnx_coord = concatenate([input_points, np_array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = concatenate([input_labels, np_array([-1])], axis=0)[None, :].astype(float32)

        onnx_coord = apply_coords(onnx_coord, embedding)
        orig_width, orig_height = embedding["original_size"]
        app_logger.info(f"onnx_coord:{onnx_coord}.")

        # RUN DECODER TO GET MASK
        onnx_mask_input = zeros((1, 1, 256, 256), dtype=float32)
        onnx_has_mask_input = zeros(1, dtype=float32)
        output_masks, _, _ = self.decoder_session.run(None, {
            "image_embeddings": embedding["image_embedding"],
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np_array([orig_height, orig_width], dtype=float32)
        })
        return output_masks

    def preprocess_image(self, img: PIL_Image | ndarray) -> ndarray:
        """
        Resize image preserving aspect ratio using `output_size_target` as a long side.

        Args:
            img: input ndarray/PIL image

        Returns:
            image ndarray

        """
        from PIL import Image

        app_logger.info(f"image type:{type(img)}, shape/size:{img.size}.")
        try:
            orig_width, orig_height = img.size
        except TypeError:
            img = Image.fromarray(img)
            orig_width, orig_height = img.size

        resized_height = self.target_size
        resized_width = int(self.target_size / orig_height * orig_width)

        if orig_width > orig_height:
            resized_width = self.target_size
            resized_height = int(self.target_size / orig_width * orig_height)

        img = img.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        return img

    def padding_tensor(self, img: PIL_Image | ndarray) -> ndarray:
        """
        Pad an image ndarray/tensor to given instance self.target_size

        Args:
            img: input ndarray/PIL image

        Returns:
            image ndarray

        """
        # Prepare input tensor from image
        tensor_input = np_array(img)
        resized_width, resized_height = img.size

        # Normalize input tensor numbers
        mean = np_array([123.675, 116.28, 103.53])
        std = np_array([[58.395, 57.12, 57.375]])
        tensor_input = (tensor_input - mean) / std

        # Transpose input tensor to shape (Batch,Channels,Height,Width
        tensor_input = tensor_input.transpose(2, 0, 1)[None, :, :, :].astype(float32)

        # Make image square self.target_size x self.target_size by padding short side by zeros
        tensor_input = np_pad(tensor_input, ((0, 0), (0, 0), (0, 0), (0, self.target_size - resized_width)))
        if resized_height < resized_width:
            tensor_input = np_pad(tensor_input, ((0, 0), (0, 0), (0, self.target_size - resized_height), (0, 0)))

        return tensor_input
