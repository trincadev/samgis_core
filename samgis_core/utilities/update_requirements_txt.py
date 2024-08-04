import argparse
import sys
from pathlib import Path
import structlog

from samgis_core.utilities import session_logger

session_logger.setup_logging(json_logs=False, log_level="INFO")
logger = structlog.stdlib.get_logger(__name__)


def get_dependencies_freeze() -> dict:
    """get a 'freeze.txt'-like dict of 'name':'version' metadata about installed packages."""
    from importlib import metadata
    return {dist.metadata["name"]: dist.version for dist in metadata.distributions()}


def sanitize_path(filename: str | Path, strict_on_filename: bool = True) -> Path:
    filename = Path(filename)
    base_path = Path.cwd().resolve(strict=True)
    logger.info(f"base_path (current working folder):{base_path} ...")
    try:
        safe_path = filename.resolve(strict=strict_on_filename)
    except FileNotFoundError as fnfe:
        logger.error(f"filename not found:{filename.absolute()}.")
        logger.error(f"fnfe:{fnfe}.")
        raise fnfe
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
    logger.debug(f"requirements_no_versions_filename:{requirements_no_versions_filename}.")
    requirements_no_versions_filename_sanitized = sanitize_path(requirements_no_versions_filename)
    logger.info(f"requirements_no_versions_filename_sanitized:{requirements_no_versions_filename_sanitized}.")
    with open(requirements_no_versions_filename_sanitized) as req_src:
        packages_no_requirements = req_src.read().split("\n")
        requirements_output = {
            name: version for name, version in sorted(freeze_dict.items()) if name in packages_no_requirements
        }
    logger.debug(f"requirements to write:{requirements_output}.")
    requirements_output_filename_sanitized = sanitize_path(requirements_output_filename, strict_on_filename=False)
    logger.debug(f"requirements_output_filename_sanitized:{requirements_output_filename_sanitized}.")
    with open(requirements_output_filename_sanitized, "w") as dst:
        out = ""
        for name, version in requirements_output.items():
            out += f"{name}=={version}\n"
        logger.info(f"output requirements content:\n{out}!")
        dst.write(out)
    logger.info(f"written requirements to file:{requirements_output_filename}!")


def get_args(current_args: list) -> argparse.Namespace:
    warning = "This file must be within the current folder."
    logger.info(f"current_args:{current_args}.")
    parser = argparse.ArgumentParser(description="Update requirements.txt from current installed packages.")
    parser.add_argument(
        "--req_no_version_path", required=True,
        help=f"file path for requirements list packages without versions. {warning}."
    )
    parser.add_argument(
        "--req_output_path", required=True,
        help=f"file path for output requirements. {warning}."
    )
    parser.add_argument(
        "--loglevel", required=False,
        default="INFO",
        choices=["DEBUG", "INFO"],
        help=f"log level (default INFO)."
    )
    args = parser.parse_args(current_args)
    logger.debug(f"args:{args}.")
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    session_logger.setup_logging(json_logs=False, log_level=args.loglevel)
    get_requirements_txt(
        requirements_no_versions_filename=args.req_no_version_path,
        requirements_output_filename=args.req_output_path
    )
