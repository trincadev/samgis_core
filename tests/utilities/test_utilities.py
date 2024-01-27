import unittest

import numpy as np


class TestUtilities(unittest.TestCase):
    def test_hash_calculate(self):
        from samgis_core.utilities.utilities import hash_calculate

        size = 5
        input_arr = np.arange(size**2).reshape((size, size))
        hash_output = hash_calculate(input_arr)
        self.assertEqual(hash_output, b'KgoWp86FwhH2tuinWOfsCfn9d+Iw6B10wwqFfdUeLeY=')

        hash_output = hash_calculate({"arr": input_arr})
        self.assertEqual(hash_output, b'M/EYsBPRQLVP9T299xHyOrtT7bdCkIDaMmW2hslMays=')

        hash_output = hash_calculate("a test string...")
        self.assertEqual(hash_output, b'29a8JwujQklQ6MKQhPyix6G1i/7Pp0uUg5wFybKuCC0=')

        hash_output = hash_calculate("123123123")
        self.assertEqual(hash_output, b'ky88G1YlfOhTmsJp16q0JVDaz4gY0HXwvfGZBWKq4+8=')

        hash_output = hash_calculate(b"a byte test string...")
        self.assertEqual(hash_output, b'dgSt/jiqLk0HJ09Xqe/BWzMvnYiOqzWlcSCCfN767zA=')

        with self.assertRaises(ValueError):
            try:
                hash_calculate(1)
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
