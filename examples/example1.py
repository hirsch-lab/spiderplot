"""
Examples demonstrating the support for categorical and numerical x-data.
"""

import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spider_chart import spiderplot, spiderplot_facet
from commons import save_figure


def example_categorical():
    """
    Basic example to create a spiderplot for categorical x-data.
    """
    k = 26
    rnd = np.random.RandomState(42)
    x = list(string.ascii_uppercase[:k])
    y = rnd.normal(loc=1, scale=1, size=k)
    s = pd.Series(y, index=x)
    df = pd.DataFrame(list(zip(x,y)), columns=["x", "y"])

    plt.figure()
    ax = spiderplot(x="x", y="y", data=df)
    ax.set_title("Example: categorical data")
    save_figure("example1/categorical.png")


def example_numeric():
    """
    Example how to use non-categorical x-data. Note: A spider plot is not
    the same as a polar plot. Both use the polar coordinate system. But
    the spider plot maps the x-data such that it spans the entire 360° on
    the (circular) x-axis (theta-axis).
    """
    n = 100
    x = np.linspace(1, 10, n)
    y = np.column_stack([np.log(x), 0.5*np.log(x)])

    plt.figure()
    ax = spiderplot(x=x, y=y,
                    hue="columns",
                    is_categorical=False,
                    n_ticks_hint=18)
    ax.set_title("Example: numerical data (is_categorical=False)")
    ax.legend(["large", "small"],
              loc="upper left",
              bbox_to_anchor=(1.1, 1.0),
              borderaxespad=0.)
    save_figure("example1/numerical.png")


if __name__ == "__main__":
    example_categorical()
    example_numeric()
    plt.show()
