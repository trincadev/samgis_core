import unittest

import numpy as np

from tests.prediction_api import helper_assertions


class TestUtilities(unittest.TestCase):
    def test_hash_calculate(self):
        from samgis_core.utilities.utilities import hash_calculate

        size = 5
        input_arr = np.arange(size**2).reshape((size, size))
        hash_output = hash_calculate(input_arr, is_file=False)
        self.assertEqual(hash_output, b'KgoWp86FwhH2tuinWOfsCfn9d+Iw6B10wwqFfdUeLeY=')

        hash_output = hash_calculate({"arr": input_arr}, is_file=False)
        self.assertEqual(hash_output, b'M/EYsBPRQLVP9T299xHyOrtT7bdCkIDaMmW2hslMays=')

        hash_output = hash_calculate("a test string...", is_file=False)
        self.assertEqual(hash_output, b'29a8JwujQklQ6MKQhPyix6G1i/7Pp0uUg5wFybKuCC0=')

        hash_output = hash_calculate("123123123", is_file=False)
        self.assertEqual(hash_output, b'ky88G1YlfOhTmsJp16q0JVDaz4gY0HXwvfGZBWKq4+8=')

        hash_output = hash_calculate(b"a byte test string...", is_file=False)
        self.assertEqual(hash_output, b'dgSt/jiqLk0HJ09Xqe/BWzMvnYiOqzWlcSCCfN767zA=')

        with self.assertRaises(ValueError):
            try:
                hash_calculate(1, is_file=False)
            except ValueError as ve:
                self.assertEqual(str(ve), "variable 'arr':1 of type '<class 'int'>' not yet handled.")
                raise ve

    def test_base64_encode_decode(self):
        from samgis_core.utilities.utilities import base64_decode, base64_encode

        dict_decode_encode = {
            "": [b"", ""],
            "hello": [b"aGVsbG8=", "hello"],
            "aGVsbG8=": [b"YUdWc2JHOD0=", "hello"]
        }
        for input_string, expected_output_list in dict_decode_encode.items():
            encoded_output = base64_encode(input_string)
            self.assertEqual(encoded_output, expected_output_list[0])

            input_bytes1 = bytes(input_string.encode('utf-8'))
            encoded_output = base64_encode(input_bytes1)
            self.assertEqual(encoded_output, expected_output_list[0])

            decoded_output = base64_decode(encoded_output)
            self.assertEqual(decoded_output, expected_output_list[0])

            decoded_output = base64_decode(input_string)
            self.assertEqual(decoded_output, expected_output_list[1])

        with self.assertRaises(ValueError):
            try:
                base64_encode(1)
            except ValueError as ve:
                self.assertEqual(str(ve), "Argument must be string or bytes")
                raise ve

    def test_normalize_array(self):
        from samgis_core.utilities.utilities import normalize_array, hash_calculate

        arr = np.arange(-50, 50).reshape(10, 10)
        arr[4:7, 2:8] = 89
        normalized = normalize_array(arr)
        hash_img = hash_calculate(normalized, is_file=False)
        helper_assertions.check_hash(hash_img, b'UdCAAQI/QcfLG8Hzgivf7FPSegNwQSXEaXX5d0Lg1Z0=')
        normalized = normalize_array(arr, new_h=1)
        hash_normalized = hash_calculate(normalized, is_file=False)
        helper_assertions.check_hash(hash_normalized, b'CZsH43+hgXZjXTqhzW7Rv4Qd93eHfd7QU7BnObmZUsc=')
        normalized = normalize_array(arr, new_h=128., type_normalization="float")
        hash_normalized = hash_calculate(normalized, is_file=False)
        helper_assertions.check_hash(hash_normalized, b'+HYPe8utlYKRqrizYPdZUINuIPqv0cIWI1zKa4tscno=')
