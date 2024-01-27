"""Get machine learning predictions from geodata raster images"""
from pathlib import Path

from samgis_core.utilities.fastapi_logger import setup_logging


app_logger = setup_logging(debug=True)
PROJECT_ROOT_FOLDER = Path(globals().get("__file__", "./_")).absolute().parent.parent
MODEL_FOLDER = Path(PROJECT_ROOT_FOLDER / "machine_learning_models")
