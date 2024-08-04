import argparse
import contextlib
import io
import os
import pathlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import structlog

from samgis_core.utilities import update_requirements_txt, session_logger


session_logger.setup_logging(json_logs=False, log_level="INFO")
test_logger = structlog.stdlib.get_logger(__name__)
freeze_mocked = {
    'flatbuffers': '24.3.25', 'coloredlogs': '15.0.1', 'humanfriendly': '10.0', 'kiwisolver': '1.4.5',
    'coverage': '7.6.0', 'pytest-cov': '5.0.0', 'Jinja2': '3.1.4', 'structlog': '24.4.0', 'six': '1.16.0',
    'attrs': '23.2.0', 'packaging': '24.1', 'iniconfig': '2.0.0', 'numpy': '1.26.4', 'pytest': '8.3.2',
    'snuggs': '1.4.7', 'pyparsing': '3.1.2', 'contourpy': '1.2.1', 'rasterio': '1.3.10', 'python-dotenv': '1.0.1',
    'python-dateutil': '2.9.0.post0', 'mpmath': '1.3.0', 'affine': '2.4.0', 'onnxruntime': '1.18.1', 'cligj': '0.7.2',
    'certifi': '2024.7.4', 'cycler': '0.12.1', 'pip': '24.2', 'pluggy': '1.5.0', 'click-plugins': '1.1.1',
    'mpld3': '0.5.10', 'samgis_core': '3.0.8', 'matplotlib': '3.9.1', 'MarkupSafe': '2.1.5', 'pillow': '10.4.0',
    'protobuf': '5.27.2', 'sympy': '1.13.1', 'fonttools': '4.53.1', 'bson': '0.5.10', 'setuptools': '72.1.0',
    'click': '8.1.7', 'ordered-set': '4.1.0', 'jaraco.text': '3.12.1', 'importlib_metadata': '8.0.0',
    'jaraco.functools': '4.0.1', 'jaraco.context': '5.3.0', 'inflect': '7.3.1', 'autocommand': '2.2.2',
    'backports.tarfile': '1.2.0', 'typeguard': '4.3.0', 'wheel': '0.43.0', 'zipp': '3.19.2',
    'importlib_resources': '6.4.0', 'tomli': '2.0.1', 'typing_extensions': '4.12.2',
    'more-itertools': '10.3.0', 'platformdirs': '4.2.2'
}
expected_requirements_txt = """bson==0.5.10
numpy==1.26.4
onnxruntime==1.18.1
pillow==10.4.0
python-dotenv==1.0.1
structlog==24.4.0
"""


@contextlib.contextmanager
def captured_output():
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class UpdateRequirementsTxt(unittest.TestCase):
    @patch.object(update_requirements_txt, "get_dependencies_freeze")
    @patch.object(update_requirements_txt, "sanitize_path")
    def test_get_requirements_txt(self, sanitize_path_mocked, get_dependencies_freeze_mocked):
        get_dependencies_freeze_mocked.return_value = freeze_mocked
        dst = Path(__file__).parent / "requirements.txt"
        requirements_no_version = Path(__file__).parent / "requirements_no_version.txt"
        # handle multiple different return values for different times sanitize_path() is called
        sanitize_path_mocked.side_effect = [requirements_no_version, dst]
        test_logger.info(f"requirements_no_version:{requirements_no_version}!")
        test_logger.info(f"file exists:{requirements_no_version.is_file()}!")
        update_requirements_txt.get_requirements_txt(requirements_no_version, dst)
        with open(dst) as check_src:
            written_requirements_content = check_src.read()
            self.assertEqual(written_requirements_content, expected_requirements_txt)
        dst.unlink()

    @patch.object(update_requirements_txt, "get_dependencies_freeze")
    def test_get_requirements_txt_traversalpath_ex(self, get_dependencies_freeze_mocked):
        get_dependencies_freeze_mocked.return_value = freeze_mocked
        with self.assertRaises(OSError):
            with tempfile.NamedTemporaryFile(prefix="../requirements_", suffix=".txt", dir=Path.cwd()) as dst:
                try:
                    requirements_no_version = Path(__file__).parent / "requirements_no_version.txt"
                    update_requirements_txt.get_requirements_txt(requirements_no_version, dst.name)
                except OSError as ose:
                    base_path = Path.cwd()
                    str_ose = str(ose).replace("  ", "").replace("\n", "")
                    # traversal path exception
                    self.assertEqual(str_ose[:26], "Basename of resolved path ")
                    self.assertIn(str(dst.name), str_ose)
                    self.assertIn(str(base_path), str_ose)
                    raise ose

    @mock.patch.object(argparse.ArgumentParser, 'parse_args')
    def test_get_args(self, parse_args_mocked):
        from samgis_core.utilities.update_requirements_txt import get_args

        parse_args_mocked.return_value = argparse.Namespace(
            requirements_no_versions_filename="req_input.txt",
            requirements_output_filename="req_output.txt"
        )
        args = get_args([])
        self.assertEqual(vars(args), {
            'requirements_no_versions_filename': 'req_input.txt',
            'requirements_output_filename': 'req_output.txt'
        })

    def test_get_args_exception(self):
        from samgis_core.utilities.update_requirements_txt import get_args
        with captured_output() as (out, err), self.assertRaises(SystemExit):
            try:
                get_args([])
            except SystemExit as sysexit:
                out.seek(0)
                self.assertEqual(out.read(), '')
                err.seek(0)
                err_content = err.read()
                self.assertIn('--req_no_version_path', err_content)
                self.assertIn('--req_output_path', err_content)
                self.assertEqual(sysexit.code, 2)
                raise sysexit
