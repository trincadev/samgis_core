"""Project constants"""
import os

DEFAULT_INPUT_SHAPE = 684, 1024
MODEL_ENCODER_NAME = os.getenv("MODEL_ENCODER_NAME", "mobile_sam.encoder.onnx")
MODEL_DECODER_NAME = os.getenv("MODEL_DECODER_NAME", "mobile_sam.decoder.onnx")
