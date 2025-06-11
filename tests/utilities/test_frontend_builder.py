import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samgis_core import app_logger
from samgis_core.utilities import frontend_builder
from tests import TEST_ROOT_FOLDER


static_folder = TEST_ROOT_FOLDER / "static"
static_css_path = str(static_folder / "src" / "input.css")
home = os.getenv("HOME")
tmp = Path(tempfile.gettempdir())
nvm_dir_mocked = tmp / ".nvm"
node_dir_parent_mocked = nvm_dir_mocked /"versions" / "node"
node_dir_mocked = node_dir_parent_mocked / "v22.12.0"
node_dir_mocked_bin = node_dir_mocked / "bin"


class TestGetNodeDirFolder(unittest.TestCase):
    def setUp(self):
        node_dir_mocked_bin.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        node_dir_mocked_bin.rmdir()
        node_dir_mocked.rmdir()
        node_dir_parent_mocked.rmdir()
        (nvm_dir_mocked /"versions").rmdir()
        nvm_dir_mocked.rmdir()

    @mock.patch.dict(os.environ, {"NODE_DIR": str(node_dir_mocked)})
    def test_get_installed_node_dir_ok(self):
        node_dir = frontend_builder.get_installed_node()
        app_logger.info(f"node_dir with NODE_DIR:{node_dir}.")
        assert str(node_dir) == str(node_dir_mocked_bin)

    @mock.patch.dict(os.environ, {"NODE_DIR_PARENT": str(node_dir_parent_mocked)})
    def test_get_installed_node_dir_parent_ok(self):
        node_dir = frontend_builder.get_installed_node()
        app_logger.info(f"node_dir with NODE_DIR_PARENT:{node_dir}.")
        assert str(node_dir) == str(node_dir_mocked_bin)

    @mock.patch.dict(os.environ, {"NODE_DIR_PARENT": "", "NODE_DIR": ""})
    def test_get_installed_node_no_env(self):
        with self.assertRaises(AssertionError):
            try:
                frontend_builder.get_installed_node()
            except AssertionError as ae:
                assert "NODE_DIR_PARENT/NODE_DIR env variable not found." in str(ae)
                raise ae

class TestGetPathWithNodeDir(unittest.TestCase):
    def setUp(self):
        node_dir_mocked_bin.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        node_dir_mocked_bin.rmdir()
        node_dir_mocked.rmdir()
        node_dir_parent_mocked.rmdir()
        (nvm_dir_mocked /"versions").rmdir()
        nvm_dir_mocked.rmdir()

    @mock.patch.dict(os.environ, {"PATH": "", "NODE_DIR": str(node_dir_mocked)})
    def test_get_path_with_node_dir_path_none(self):
        path = frontend_builder.get_path_with_node_dir()
        assert f"{node_dir_mocked_bin}:" == path

    @mock.patch.dict(os.environ, {"PATH": "/bin:", "NODE_DIR": str(node_dir_mocked)})
    def test_get_path_with_node_dir_path_no_node(self):
        path = frontend_builder.get_path_with_node_dir()
        assert f"{node_dir_mocked_bin}:/bin:" == path

    @mock.patch.dict(os.environ, {"PATH": f"{node_dir_mocked_bin}:/bin:", "NODE_DIR": str(node_dir_mocked)})
    def test_get_path_with_node_dir_path_no_node__node_dir_mocked_bin_bin(self):
        path = frontend_builder.get_path_with_node_dir()
        app_logger.info(f"PATH '{path}'.")
        assert f"{node_dir_mocked_bin}:/bin:" == path


class TestFrontendBuilder(unittest.TestCase):
    def test_frontend_builder(self):
        import shutil

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


if __name__ == '__main__':
    unittest.main()
    print("all ok!")
