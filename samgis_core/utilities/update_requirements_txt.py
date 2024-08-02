from pathlib import Path
import structlog

from samgis_core.utilities import session_logger


session_logger.setup_logging(json_logs=False, log_level="INFO")
logger = structlog.stdlib.get_logger(__name__)


def get_dependencies_freeze() -> dict:
    """get a 'freeze.txt'-like dict of 'name':'version' metadata about installed packages."""
    from importlib import metadata
    return {dist.metadata["name"]: dist.version for dist in metadata.distributions()}


def get_requirements_txt(requirements_no_versions_filename: str | Path, requirements_output_filename: str | Path):
    """
    Write on disk a requirements.txt file with an updated list of dependencies from installed python packages.

    Args:
        requirements_no_versions_filename: full path of input requirements filename with no specified versions
        requirements_output_filename: full path of output requirements.txt filename

    Returns:

    """
    logger.info("start requirements.txt update...")
    freeze_dict = get_dependencies_freeze()
    logger.debug(f"freeze_dict:{freeze_dict}.")
    with open(requirements_no_versions_filename) as req_src:
        packages_no_requirements = req_src.read().split("\n")
        requirements_output = {
            name: version for name, version in sorted(freeze_dict.items()) if name in packages_no_requirements
        }
    logger.info(f"requirements to write:{requirements_output}.")
    with open(requirements_output_filename, "w") as dst:
        out = ""
        for name, version in requirements_output.items():
            out += f"{name}=={version}\n"
        logger.debug(f"output requirements content:\n{out}!")
        dst.write(out)
    logger.info(f"written requirements to file:{requirements_output_filename}!")
