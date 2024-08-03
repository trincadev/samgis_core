from pathlib import Path
import structlog

from samgis_core.utilities import session_logger

session_logger.setup_logging(json_logs=False, log_level="INFO")
logger = structlog.stdlib.get_logger(__name__)


def get_dependencies_freeze() -> dict:
    """get a 'freeze.txt'-like dict of 'name':'version' metadata about installed packages."""
    from importlib import metadata
    return {dist.metadata["name"]: dist.version for dist in metadata.distributions()}


def sanitize_path(filename: str | Path) -> Path:
    filename = Path(filename)
    base_path = Path(__file__).parent.resolve(strict=True)
    safe_path = filename.resolve(strict=True)
    try:
        assert base_path / filename.name == safe_path
    except AssertionError:
        msg = f"""
        Basename of resolved path {safe_path} doesn't matches original file {filename},
        or filename {filename} isn't within the current directory {base_path} ...
        """
        raise OSError(msg)
    return Path(safe_path)


def get_requirements_txt(requirements_no_versions_filename: str | Path, requirements_output_filename: str | Path):
    """
    Write on disk a requirements.txt file with an updated list of dependencies from installed python packages.
    Both input and output requirements files must be within the folder from which the current command is executed.

    Args:
        requirements_no_versions_filename: input requirements filename with no specified versions
        requirements_output_filename: output requirements.txt filename

    Returns:

    """
    logger.info("start requirements.txt update...")
    freeze_dict = get_dependencies_freeze()
    logger.debug(f"freeze_dict:{freeze_dict}.")
    requirements_no_versions_filename = sanitize_path(requirements_no_versions_filename)
    with open(requirements_no_versions_filename) as req_src:
        packages_no_requirements = req_src.read().split("\n")
        requirements_output = {
            name: version for name, version in sorted(freeze_dict.items()) if name in packages_no_requirements
        }
    logger.info(f"requirements to write:{requirements_output}.")
    requirements_output_filename = sanitize_path(requirements_output_filename)
    with open(requirements_output_filename, "w") as dst:
        out = ""
        for name, version in requirements_output.items():
            out += f"{name}=={version}\n"
        logger.info(f"output requirements content:\n{out}!")
        dst.write(out)
    logger.info(f"written requirements to file:{requirements_output_filename}!")


if __name__ == '__main__':
    import argparse
    from pathvalidate.argparse import sanitize_filepath_arg

    warning = "This file must be within the current folder."
    parser = argparse.ArgumentParser(description="Update requirements.txt from current installed packages.")
    parser.add_argument(
        "--req_no_version_path", required=True,
        type=sanitize_filepath_arg,
        help=f"file path for requirements list packages without versions. {warning}."
    )
    parser.add_argument(
        "--req_output_path", required=True,
        type=sanitize_filepath_arg,
        help=f"file path for output requirements. {warning}."
    )
    args = parser.parse_args()
    get_requirements_txt(
        requirements_no_versions_filename=args.req_no_version_path,
        requirements_output_filename=args.req_output_path
    )
