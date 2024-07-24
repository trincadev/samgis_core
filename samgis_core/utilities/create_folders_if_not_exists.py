import json
import logging
import os
from pathlib import Path


def stats_pathname(pathname: Path | str):
    current_pathname = Path(pathname)
    return current_pathname.is_dir()


def create_folder_if_not_exists(pathname: Path | str):
    current_pathname = Path(pathname)
    try:
        print(f"Pathname exists? {current_pathname.exists()}, That's a folder? {current_pathname.is_dir()}...")
        logging.info(f"Pathname exists? {current_pathname.exists()}, That's a folder? {current_pathname.is_dir()}...")
        current_pathname.unlink(missing_ok=True)
    except PermissionError as pe:
        print(f"permission denied on removing pathname before folder creation:{pe}.")
        logging.error(f"permission denied on removing pathname before folder creation:{pe}.")
    except IsADirectoryError as errdir:
        print(f"that's a directory:{errdir}.")
        logging.error(f"that's a directory:{errdir}.")

    print(f"Creating pathname: {current_pathname} ...")
    logging.info(f"Creating pathname: {current_pathname} ...")
    current_pathname.mkdir(mode=0o770, parents=True, exist_ok=True)

    print(f"assertion: pathname exists and is a folder: {current_pathname} ...")
    logging.info(f"assertion: pathname exists and is a folder: {current_pathname} ...")
    assert current_pathname.is_dir()


def folders_creation(folders_map: dict | str = None, ignore_errors: bool = True):
    enforce_validation_with_getenv = folders_map is None
    if enforce_validation_with_getenv:
        folders_map = os.getenv("FOLDERS_MAP")
    try:
        folders_dict = folders_map if isinstance(folders_map, dict) else json.loads(folders_map)
        for folder_env_ref, folder_env_path in folders_dict.items():
            logging.info(f"folder_env_ref:{folder_env_ref}, folder_env_path:{folder_env_path}.")
            create_folder_if_not_exists(folder_env_path)
            print("========")
            if enforce_validation_with_getenv:
                assert os.getenv(folder_env_ref) == folder_env_path
    except (json.JSONDecodeError, TypeError) as jde:
        logging.error(f"jde:{jde}.")
        msg = "double check your variables, e.g. for misspelling like 'FOLDER_MAP'"
        msg += "instead than 'FOLDERS_MAP', or invalid json values."
        logging.error(msg)
        for k_env, v_env in dict(os.environ).items():
            logging.info(f"{k_env}, v_env:{v_env}.")
        if not ignore_errors:
            raise TypeError(jde)


if __name__ == '__main__':
    folders_creation()

