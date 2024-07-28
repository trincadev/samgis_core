import json
import logging
import os
from pathlib import Path


def stats_pathname(pathname: Path | str):
    current_pathname = Path(pathname)
    return current_pathname.is_dir()


def create_folder_if_not_exists(pathname: Path | str):
    """Create a folder given its path.

    Args:
        pathname: folder Path or string

    Returns:

    """
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
    """Create all folders listed within the folders_map argument (this argument can be a dict or a json string).
    If folders_map is None the function will try to load the 'FOLDERS_MAP' env variable, then will load that json into
    dict. Once loaded and parsed the folders_map variable, the function will loop over the dict to create the folders
    using the `create_folder_if_not_exists()` function.

    Args:
        folders_map: dict or string map of folder string
        ignore_errors: bool needed to eventually ignore errors on folder creation

    Returns:

    """
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

