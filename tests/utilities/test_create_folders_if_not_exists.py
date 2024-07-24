import json
import logging
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from samgis_core.utilities import create_folders_if_not_exists


tmp = tempfile.gettempdir()
folders_map = {f"folder_{n}": str(Path(tmp) / f"{random.random()}") for n in range(3)}
env_dict = {
    "FOLDERS_MAP": json.dumps(folders_map),
    **folders_map
}


class TestCreateFoldersIfNotExists(unittest.TestCase):
    def test_stats_pathname(self):

        with tempfile.TemporaryDirectory() as tmp_dir:
            assert create_folders_if_not_exists.stats_pathname(tmp_dir)

    def test_create_folder_if_not_exists(self):
        tmp_subfolder = Path(tmp) / f"{random.random()}"
        create_folders_if_not_exists.create_folder_if_not_exists(tmp_subfolder)
        shutil.rmtree(tmp_subfolder)

    def test_create_folder_if_not_exists_error1(self):
        with self.assertRaises(FileExistsError):
            try:
                create_folders_if_not_exists.create_folder_if_not_exists("/dev/null")
            except FileExistsError as fe_error:
                msg = str(fe_error)
                self.assertEquals(msg, "[Errno 17] File exists: '/dev/null'")
                raise FileExistsError

    def test_folders_creation_map_as_argument(self):
        create_folders_if_not_exists.folders_creation(folders_map)
        for folder_name, folder_path in folders_map.items():
            try:
                shutil.rmtree(folder_path)
            except Exception as ex:
                logging.error(f"error on removing folder {folder_name} => {folder_path}: {ex}.")
                raise ex

    @mock.patch.dict(os.environ, env_dict)
    def test_folders_creation_map_as_env(self):
        create_folders_if_not_exists.folders_creation()
        for folder_name, folder_path in folders_map.items():
            try:
                shutil.rmtree(folder_path)
            except Exception as ex:
                logging.error(f"error on removing folder {folder_name} => {folder_path}: {ex}.")
                raise ex

    @mock.patch.dict(os.environ, {"FOLDERS_MAP": ""})
    def test_create_folder_if_not_exists_json_decode_error(self):
        with self.assertRaises(TypeError):
            try:
                create_folders_if_not_exists.folders_creation(ignore_errors=False)
            except TypeError as json_type_error:
                msg = str(json_type_error)
                self.assertEquals(msg, 'Expecting value: line 1 column 1 (char 0)')
                raise json_type_error
