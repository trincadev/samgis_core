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
        if str(node_dir) != str(node_dir_mocked_bin):
            raise ValueError(f"wrong node folders: get_installed_node '{node_dir}' vs mocked '{node_dir_mocked_bin}' #")

    @mock.patch.dict(os.environ, {"NODE_DIR_PARENT": str(node_dir_parent_mocked)})
    def test_get_installed_node_dir_parent_ok(self):
        node_dir = frontend_builder.get_installed_node()
        app_logger.info(f"node_dir with NODE_DIR_PARENT:{node_dir}.")
        if str(node_dir) != str(node_dir_mocked_bin):
            raise ValueError(f"wrong node folders: get_installed_node '{node_dir}' vs mocked '{node_dir_mocked_bin}' #")

    @mock.patch.dict(os.environ, {"NODE_DIR_PARENT": "", "NODE_DIR": ""})
    def test_get_installed_node_no_env(self):
        with self.assertRaises(ValueError):
            try:
                frontend_builder.get_installed_node()
            except ValueError as ve:
                if "NODE_DIR_PARENT/NODE_DIR env variable not found." not in str(ve):
                    raise AssertionError("ValueError exception hasn't the right message!")
                raise ve

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
        if f"{node_dir_mocked_bin}:" != path:
            raise ValueError(f"wrong folders: {node_dir_mocked_bin} not in {path} #")

    @mock.patch.dict(os.environ, {"PATH": "/bin:", "NODE_DIR": str(node_dir_mocked)})
    def test_get_path_with_node_dir_path_no_node(self):
        path = frontend_builder.get_path_with_node_dir()
        if f"{node_dir_mocked_bin}:/bin:" != path:
            raise ValueError(f"wrong folders: {node_dir_mocked_bin}:/bin: not in {path} #")

    @mock.patch.dict(os.environ, {"PATH": f"{node_dir_mocked_bin}:/bin:", "NODE_DIR": str(node_dir_mocked)})
    def test_get_path_with_node_dir_path_no_node__node_dir_mocked_bin_bin(self):
        path = frontend_builder.get_path_with_node_dir()
        app_logger.info(f"PATH '{path}'.")
        if f"{node_dir_mocked_bin}:/bin:" != path:
            raise ValueError(f"wrong folders: {node_dir_mocked_bin}:/bin: not in {path} #")


class TestFrontendBuilder(unittest.TestCase):
    def test_frontend_builder(self):
        import shutil

        static_dist_folder = static_folder / "dist"
        if not frontend_builder.build_frontend(
                project_root_folder=TEST_ROOT_FOLDER,
                input_css_path=static_css_path,
                output_dist_folder=static_dist_folder,
                force_build=True
            ):
            raise ValueError("Frontend build failed/1")
        if not static_dist_folder.is_dir():
            raise FileNotFoundError(f"static_dist_folder folder not found: {static_dist_folder} #")
        index_html = static_dist_folder / "index.html"
        output_css = static_dist_folder / "output.css"
        if not index_html.is_file():
            raise ValueError(f"missing file: {index_html} #")
        if not output_css.is_file():
            raise ValueError(f"missing file: {output_css} #")
        with open(index_html, "r") as html_src:
            html_body = html_src.read()
            if not ("html" in html_body and "head" in html_body and "body" in html_body):
                raise ValueError("Wrong frontend build: index html malfolded!")
        if output_css.stat().st_size < 1:
            raise IOError("empty output_css file")
        if frontend_builder.build_frontend(
                project_root_folder=TEST_ROOT_FOLDER,
                input_css_path=static_css_path,
                output_dist_folder=static_dist_folder
            ):
            raise ValueError("wrong frontend build/2")
        if frontend_builder.build_frontend(
                project_root_folder=TEST_ROOT_FOLDER,
                input_css_path=static_css_path,
                output_dist_folder=static_dist_folder,
                force_build=False
            ):
            raise ValueError("wrong frontend build/3")
        shutil.rmtree(static_dist_folder, ignore_errors=False)


if __name__ == '__main__':
    unittest.main()
    print("all ok!")
