import importlib.metadata


try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except Exception:
    __version__ = importlib_metadata.version(__package__ or __name__)
