import os
from pathlib import Path
import unittest
from unittest import mock

from tests import TEST_ROOT_FOLDER


static_folder = TEST_ROOT_FOLDER / "static"
static_css_path = str(static_folder / "src" / "input.css")


class TestFrontendBuilder(unittest.TestCase):
    @mock.patch.dict(os.environ, {"INPUT_CSS_PATH": static_css_path})
    def test_frontend_builder(self):
        import shutil
        from samgis_core.utilities import frontend_builder

        static_dist_folder = static_folder / "dist"
        assert frontend_builder.build_frontend(
            project_root_folder=TEST_ROOT_FOLDER,
            input_css_path=static_css_path,
            output_dist_folder=static_dist_folder,
            force_build=True
        )
        assert static_dist_folder.is_dir()
        index_html = static_dist_folder / "index.html"
        output_css = static_dist_folder / "output.css"
        assert index_html.is_file()
        assert output_css.is_file()
        with open(index_html, "r") as html_src:
            html_body = html_src.read()
            assert "html" in html_body
            assert "head" in html_body
            assert "body" in html_body
        assert output_css.stat().st_size > 0
        assert not frontend_builder.build_frontend(
            project_root_folder=TEST_ROOT_FOLDER,
            input_css_path=static_css_path,
            output_dist_folder=static_dist_folder
        )
        assert not frontend_builder.build_frontend(
            project_root_folder=TEST_ROOT_FOLDER,
            input_css_path=static_css_path,
            output_dist_folder=static_dist_folder,
            force_build=False
        )
        shutil.rmtree(static_dist_folder, ignore_errors=False)
