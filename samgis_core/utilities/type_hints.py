"""custom type hints"""
from enum import Enum
from typing import TypedDict, NewType

from PIL.Image import Image
from numpy import ndarray


class ListStr(list[str]): pass


class DictStrInt(dict[str, int]): pass


class DictStr(dict[str]): pass


class DictStrAny(dict[str, any]): pass


class ListDict(list[dict]): pass


class ListFloat(list[float]): pass


class ListInt(list[int]): pass


class TupleInt(tuple[int]): pass


TupleInt2 = NewType("TupleInt", tuple[int, int])


class TupleNdarrayInt(tuple[ndarray, int]): pass


class TupleNdarrayFloat(tuple[ndarray, float]): pass


class LlistFloat(ListFloat): pass


class TupleFloat(tuple[float]): pass


class TupleFloatAny(tuple[float, any]): pass


PIL_Image = Image


class StrEnum(str, Enum):
    pass


class EmbeddingImage(TypedDict):
    image_embedding: ndarray
    original_size: TupleInt
    transform_matrix: ndarray


class EmbeddingPILImage(TypedDict):
    image_embedding: ndarray
    original_size: TupleInt2
    resized_size: TupleInt2


class MatplotlibBackend(StrEnum):
    gtk3agg = "gtk3agg"
    gtk3cairo = "gtk3cairo"
    gtk4agg = "gtk4agg"
    gtk4cairo = "gtk4cairo"
    macosx = "macosx"
    nbagg = "nbagg"
    notebook = "notebook"
    qtagg = "qtagg"
    qtcairo = "qtcairo"
    qt5agg = "qt5agg"
    qt5cairo = "qt5cairo"
    tkagg = "tkagg"
    tkcairo = "tkcairo"
    webagg = "webagg"
    wx = "wx"
    wxagg = "wxagg"
    wxcairo = "wxcairo"
    agg = "agg"
    cairo = "cairo"
    pdf = "pdf"
    pgf = "pgf"
    ps = "ps"
    svg = "svg"
    template = "template"


EmbeddingDict = dict[str, EmbeddingImage]
EmbeddingPILDict = dict[str, EmbeddingPILImage]
