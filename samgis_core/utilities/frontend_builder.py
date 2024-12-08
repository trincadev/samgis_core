import os
import subprocess
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from samgis_core import app_logger
from samgis_core.utilities.type_hints import ListStr


load_dotenv()
root_folder = Path(globals().get("__file__", "./_")).absolute().parent.parent.parent
env_project_root_folder = os.getenv("PROJECT_ROOT_FOLDER", str(root_folder))
env_input_css_path = os.getenv("INPUT_CSS_PATH")


def assert_envs(envs_list: ListStr) -> None:
    """
    Assert env variables are not empty.

    Args:
        envs_list: list of env variables

    Returns:

    """
    for current_env in envs_list:
        try:
            assert current_env is not None and current_env != ""
        except AssertionError as aex:
            app_logger.error(f"error on assertion for current_env: {current_env}.")
            raise aex


def read_std_out_err(std_out_err: str, output_type: str, command: ListStr) -> None:
    """
    Capture the standard output or standard error content for a given system command pipe.

    Args:
        std_out_err: str
        output_type: output type (stdout or stderr)
        command: command executed

    Returns:

    """
    output = std_out_err.split("\n")
    app_logger.info(f"output type:{output_type} for command:{' '.join(command)}.")
    for line in iter(output):
        app_logger.info(f"output_content_home stdout:{line.strip()}.")
    app_logger.info("########")


def run_command(commands_list: ListStr, capture_output: bool = True, text: bool = True, check: bool = True, env: Mapping[str, str] | None = None) -> None:
    """
    Run a system command and capture its output.
    See https://docs.python.org/3.11/library/subprocess.html#subprocess.run for more details

    Args:
        commands_list: list of single string commands
        capture_output: if true, stdout and stderr will be captured
        text: if true, capture stdout and stderr as strings
        check: If check is true, and the process exits with a non-zero exit code, a CalledProcessError exception will
               be raised. Attributes of that exception hold the arguments, the exit code, and stdout and stderr if they
               were captured.
        env: environment variables

    Returns:

    """
    try:
        output_content_home = subprocess.run(
            commands_list,
            capture_output=capture_output,
            text=text,
            check=check,
            env=env
        )
        read_std_out_err(output_content_home.stdout, "stdout", commands_list)
        read_std_out_err(output_content_home.stderr, "stderr", commands_list)
    except Exception as ex:
        app_logger.error(f"ex:{ex}.")
        raise ex


def build_frontend(
        project_root_folder: str | Path,
        input_css_path: str | Path,
        output_dist_folder: Path = root_folder / "static" / "dist",
        index_page_filename: str = "index.html",
        output_css_filename: str = "output.css",
        force_build: bool = False,
    ) -> bool:
    """
    Build a static [Vue js](https://vuejs.org/), [tailwindcss](https://tailwindcss.com/) frontend.
    If force_build is False, the function also check if index_page_filename and output_css_filename already exists:
    in this case skip the build.

    Args:
        project_root_folder: Project folder that contains the static frontend
        input_css_path: file path pointing to the input css file
        output_dist_folder: dist folder path where to write the frontend bundle
        index_page_filename: index html filename
        output_css_filename: output css filename
        force_build: useful to skip the frontend build

    Returns:
        state of the build (True in case of build completed, False in case of build skipped)

    """
    assert_envs([
        str(project_root_folder),
        str(input_css_path)
    ])
    project_root_folder = Path(project_root_folder)
    index_html_pathfile = Path(output_dist_folder) / index_page_filename
    output_css_pathfile = Path(output_dist_folder) / output_css_filename
    if not force_build and output_css_pathfile.is_file() and index_html_pathfile.is_file():
        app_logger.info("frontend ok, build_frontend not necessary...")
        return False

    # install deps
    os.chdir(project_root_folder / "static")
    current_folder = os.getcwd()
    app_logger.info(f"current_folder:{current_folder}, install pnpm...")
    run_command(["which", "npm"])
    run_command(["npm", "install", "-g", "npm", "pnpm"])
    app_logger.info("install pnpm dependencies...")
    run_command(["pnpm", "install"])

    # build frontend dist and assert for its correct build
    output_css = str(output_css_pathfile)
    output_index_html = str(output_dist_folder / index_page_filename)
    output_dist_folder = str(output_dist_folder)
    app_logger.info(f"pnpm: build '{output_dist_folder}'...")
    run_command(["pnpm", "build"])
    app_logger.info(f"pnpm: ls -l {output_index_html}:")
    run_command(["ls", "-l", output_index_html])
    cmd = ["pnpm", "tailwindcss", "-i", str(input_css_path), "-o", output_css]
    app_logger.info(f"pnpm: {' '.join(cmd)}...")
    run_command(["pnpm", "tailwindcss", "-i", str(input_css_path), "-o", output_css])
    app_logger.info(f"pnpm: ls -l {output_css}:")
    run_command(["ls", "-l", output_css])
    app_logger.info("end!")
    return True


def get_installed_node():
    """
    Get the installed node bin folder given the environment variables:
    - NODE_DIR (current node installed), without the bin subfolder
    - NODE_DIR_PARENT (parent folder of the node installed versions, e.g. `~/.nvm/versions/node/` )

    Returns:
        node bin path

    """
    node_dir = os.getenv("NODE_DIR")
    if not (node_dir and Path(node_dir).exists() and Path(node_dir).is_dir()):
        app_logger.error(f"node_dir:{node_dir} not found.")
        node_dir_env = os.getenv("NODE_DIR_PARENT")
        assert node_dir_env is not None and node_dir_env != "", "NODE_DIR_PARENT/NODE_DIR env variable not found."
        node_dir_parent = Path(os.getenv("NODE_DIR_PARENT"))
        app_logger.error(f"try with '{node_dir_parent}' ...")
        assert node_dir_parent.exists() and node_dir_parent.is_dir(), "node_dir_parent not found"
        list_node_folders = [f.path for f in os.scandir(node_dir_parent) if f.is_dir()]
        list_node_folders.sort()
        assert isinstance(list_node_folders, list) and len(list_node_folders) > 0, "no node folders found."
        node_dir = list_node_folders.pop()
    node_dir_bin = Path(node_dir) / "bin"
    assert node_dir_bin.exists() and node_dir_bin.is_dir(), f"node's bin not found or not a directory: {node_dir}."
    return node_dir_bin


def get_path_with_node_dir():
    """
    Get the PATH environment variable with the node bin folder.

    Returns:
        PATH environment variable with the node bin folder

    """
    node_dir = get_installed_node()
    path = os.getenv("PATH")
    if path is None or path == "":
        return f"{node_dir}:"
    if "node" not in path:
        return f"{node_dir}:{path}"
    return path


if __name__ == '__main__':
    build_frontend(
        project_root_folder=Path(env_project_root_folder),
        input_css_path=Path(env_input_css_path)
    )
