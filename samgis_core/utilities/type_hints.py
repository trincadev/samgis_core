"""custom type hints"""
from PIL.Image import Image
from numpy import ndarray

dict_str_int = dict[str, int]
dict_str = dict[str]
dict_str_any = dict[str, any]
list_dict = list[dict]
list_float = list[float]
list_int = list[int]
tuple_int = tuple[int]
tuple_ndarr_int = tuple[ndarray, int]
llist_float = list[list_float]
tuple_float = tuple[float]
tuple_float_any = tuple[float, any]
PIL_Image = Image
