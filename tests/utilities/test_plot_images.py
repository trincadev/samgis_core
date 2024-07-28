import tempfile
import unittest

import numpy as np
from PIL import Image

from samgis_core.utilities import plot_images
from samgis_core.utilities.utilities import hash_calculate
from tests import TEST_EVENTS_FOLDER


folder = TEST_EVENTS_FOLDER / "samexporter_predict" / "colico"


class TestPlotImages(unittest.TestCase):
    def test_helper_imshow_output_expected(self):
        img_list = [
            Image.open(folder / "colico_nextzen_rgb.png"),
            Image.open(folder / "colico_nextzen.png"),
        ]
        titles_list = ["colico_nextzen_rgb", "colico_nextzen"]
        fig, ax = plot_images.helper_imshow_output_expected(
            img_list, titles_list, show=True, close_after=0.01)  # , debug=True)
        with tempfile.NamedTemporaryFile(prefix="tmp_img_list_", suffix=".png") as tmp_file:
            fig.savefig(tmp_file.name)
            saved_img = Image.open(tmp_file.name)
            np_saved_img = np.array(saved_img)
            hash_output = hash_calculate(np_saved_img)
            assert hash_output == b'YRrEKeLZNTqxxHdzrEFpASiFQPhngRetOtDeu1D5Z8I='

    def test_imshow_raster(self):
        """supported matplotlib backend to set in plt.rcParams["backend"]:
        [
            'gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx', 'nbagg', 'notebook', 'qtagg', 'qtcairo', 'qt5agg',
            'qt5cairo', 'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps',
            'svg', 'template'
        ]
        """
        img = Image.open(folder / "colico_nextzen_rgb.png")
        img_np = np.array(img)
        fig, ax = plot_images.imshow_raster(img_np, "colico", show=True, close_after=0.01)  # , debug=True)
        with tempfile.NamedTemporaryFile(prefix="tmp_img_", suffix=".png") as tmp_file:
            fig.savefig(tmp_file.name)
            saved_img = Image.open(tmp_file.name)
            np_saved_img = np.array(saved_img)
            hash_output = hash_calculate(np_saved_img)
            assert hash_output == b'1nEoSvWN7cMRjkaw1pVF6UyygNTYWIUedlDmkKSi0eY='


if __name__ == '__main__':
    unittest.main()
