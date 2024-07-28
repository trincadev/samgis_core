import structlog.stdlib
from numpy import ndarray
from matplotlib import pyplot as plt

from samgis_core.utilities.type_hints import ListStr, MatplotlibBackend

FigAxes = tuple[plt.Figure, plt.Axes]


logger = structlog.stdlib.get_logger(__file__)


def helper_imshow_output_expected(
        img_list: list[ndarray], titles_list: ListStr, cmap: str = "gist_rainbow", plot_size: int = 5,
        show=False, debug: bool = False, close_after: float = 0.0) -> FigAxes:
    """
    Simple way to display a list of images with their titles, color map.
    Should work also in an automate environments, like tests (use a `close_after` argument > 0)

    Args:
        img_list: ndarray images to display
        titles_list: title images
        cmap: color map
        plot_size: figure plot size
        show: fire plt.show() action if needed
        debug: workaround useful in an interactive context, like Pycharm debugger
        close_after: close after give seconds (useful in tests, contrasted to 'debug' option)

    Returns:
        tuple of matplotlib Figure, Axes

    """
    n = len(img_list)
    assert len(titles_list) == n
    fig, ax = plt.subplot_mosaic([
        titles_list
    ], figsize=(n * plot_size, plot_size))

    for title, img in zip(titles_list, img_list):
        ax[title].imshow(img, cmap=cmap)
        ax[title].legend()
    if show:
        if debug:
            plt.pause(0.01)
            plt.show()
        if close_after > 0:
            plt.pause(close_after)
            plt.show(block=False)
            plt.close("all")
    return fig, ax


def imshow_raster(
        raster, title, cmap: str = "gist_rainbow", interpolation: str = None, alpha=None, transform=None, plot_size=5,
        show=False, debug: bool = False, close_after: float = 0.0, backend: MatplotlibBackend = None) -> FigAxes:
    """
    Displays raster images lists/arrays with titles, legend, alpha transparency, figure sizes
    and geographic transformations, if not none (leveraging rasterio.plot)

    Args:
        raster: image to display
        title: title image
        cmap: color map
        interpolation: interpolation type
        alpha: alpha transparency
        transform: geographic transform, eventually used for map representation by rasterio
        plot_size: figure plot size
        show: fire plt.show() action if needed
        debug: workaround useful in an interactive context, like Pycharm debugger
        close_after: close after give seconds (useful in tests, contrasted to 'debug' option)
        backend: matplotlib backend string

    Returns:
        tuple of matplotlib Figure, Axes

    """
    from rasterio import plot

    if not backend:
        backend = plt.get_backend()
        plt.rcParams["backend"] = backend
    logger.info(f"use {backend} as matplotlib backend...")

    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    raster_ax = raster[0] if transform is not None else raster
    image_hidden = ax.imshow(raster_ax, cmap=cmap, interpolation=interpolation, alpha=alpha)
    if transform is not None:
        plot.show(raster, transform=transform, ax=ax, cmap=cmap, interpolation=interpolation, alpha=alpha)
    fig.colorbar(image_hidden, ax=ax)
    ax.set_title(title)
    if show:
        if debug:
            plt.pause(0.01)
            plt.show()
        if close_after > 0:
            plt.pause(close_after)
            plt.show(block=False)
            plt.close("all")
    return fig, ax
