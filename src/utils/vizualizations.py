from numpy import ndarray
from pandas import DataFrame
from matplotlib.axes._subplots import SubplotBase


def add_text_barplot(
    ax: SubplotBase, feature: ndarray | DataFrame, x_pos_diff: int = 0.05
) -> None:
    """Simple method to add the percentage value on top of a seaborn (matplotlib) barplot.

    Parameters
    ----------
    ax : AxesSubplot
        axes to add text onto
    feature : ndarray | DataFrame
        values used to create the barplot
    """
    total = len(feature)
    for p in ax.patches:
        percentage = "{:.1f}%".format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width() / 2 - x_pos_diff
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), size=12)
