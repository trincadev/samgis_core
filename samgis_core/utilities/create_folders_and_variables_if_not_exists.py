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


def folders_creation(folders_map: dict | str = None):
    if folders_map is None:
        folders_map = os.getenv("FOLDERS_MAP")
    try:
        folders_dict = folders_map if isinstance(folders_map, dict) else json.loads(folders_map)
        for folder_env_ref, folder_env_path in folders_dict.items():
            print(f"folder_env_ref:{folder_env_ref}, folder_env_path:{folder_env_path}.")
            logging.info(f"folder_env_ref:{folder_env_ref}, folder_env_path:{folder_env_path}.")
            create_folder_if_not_exists(folder_env_path)
            print("========")
            assert os.getenv(folder_env_ref) == folder_env_path
    except (json.JSONDecodeError, TypeError) as jde:
        print(f"jde:{jde}.")
        logging.error(f"jde:{jde}.")
        print("double check your variables, e.g. for misspelling like 'FOLDER_MAP'...")
        logging.info("double check your variables, e.g. for misspelling like 'FOLDER_MAP' instead than 'FOLDERS_MAP'...")
        for k_env, v_env in dict(os.environ).items():
            print(f"{k_env}, v_env:{v_env}.")
            logging.info(f"{k_env}, v_env:{v_env}.")


if __name__ == '__main__':
    folders_creation()

