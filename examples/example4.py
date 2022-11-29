"""
Examples illustrating equivalence and difference of different modes.
"""
import string
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from spiderplot import spiderplot, spiderplot_facet
from commons import save_figure


def example_input1():
    """
    Example illustrating different input modes.
    """
    k = 12
    rnd = np.random.RandomState(42)
    x = list(string.ascii_uppercase[:k])
    y = rnd.normal(loc=1, scale=1, size=k)
    s = pd.Series(y, index=x)
    df = pd.DataFrame(list(zip(x,y)), columns=["x", "y"])
    _, axes = plt.subplots(2, 2, figsize=(6,6),
                           subplot_kw={"polar": True})
    ((ax1, ax2), (ax3, ax4)) = axes
    spiderplot(y=y, ax=ax1)
    spiderplot(x=x, y=y, ax=ax2)
    spiderplot(data=s, ax=ax3)
    spiderplot(x="x", y="y", data=df, ax=ax4)

    ax1.set_title("array y", pad=30)
    ax2.set_title("arrays x and y", pad=30)
    ax3.set_title("pd.Series(y, index=x)", pad=30)
    ax4.set_title("pd.DataFrame(zip(x,y))", pad=30)
    plt.suptitle("Input modes")
    plt.tight_layout()


def example_input2():
    k = 11
    m = 3

    rnd = np.random.RandomState(42)
    x = list(string.ascii_uppercase[:k])
    y = rnd.normal(loc=1, scale=1, size=(k, m))
    df_wide = pd.DataFrame(y, index=x, columns=["i", "ii", "iii"])
    df_wide.index.name = "x"
    df_wide.columns.name = "category"
    df = df_wide.reset_index().melt(value_name="y", id_vars="x")

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,3),
                                      subplot_kw={"polar": True})
    spiderplot(x=x, y=y, ax=ax1, hue="columns", fill=True)
    spiderplot(x="x", y="y", hue="category", data=df,
               ax=ax2, fill=True, marker="o")
    spiderplot(data=df_wide, ax=ax3, fill=True)
    ax1.set_title("Array data", pad=30)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.2, 1.0), borderaxespad=0.)
    ax2.set_title("Long-form data", pad=30)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.2, 1.0), borderaxespad=0.)
    ax3.set_title("Wide-form data", pad=30)
    ax3.legend(loc="upper left", bbox_to_anchor=(1.2, 1.0), borderaxespad=0.)
    plt.tight_layout()


def example_aggregation():
    k = 16
    n = 5

    rnd = np.random.RandomState(42)
    x = list(string.ascii_uppercase[:k])
    y = rnd.normal(loc=1, scale=0.1, size=(k, n))
    means = y.mean(axis=1)
    stds = y.std(axis=1)
    df = pd.DataFrame(list(zip(means, stds)),
                      columns=["mean", "std"],
                      index=x)
    df.index.name = "x"
    df = df.reset_index()

    _, axes = plt.subplots(2, 2, figsize=(6,6),
                           subplot_kw={"polar": True})
    ((ax1, ax2), (ax3, ax4)) = axes
    # Pass mean y-data and extent.
    spiderplot(y=means, extent=stds, fill=False, ax=ax1)
    # Pass mean xy-data and extent.
    spiderplot(x=x, y=means, extent=stds, fill=False, ax=ax2)
    # Pass mean xy-data and extent as table.
    spiderplot(x="x", y="mean", extent="std", data=df, fill=False, ax=ax3)
    # Pass non-aggregated xy-data, using the aggregation capabilities
    # of sns.lineplot()). Note that the "closure" is missing here because
    # sns.lineplot() is unaware of the cyclic nature of a spiderplot().
    spiderplot(x=x, y=y, fill=False, ax=ax4)
    ax1.set_title("mean±std (array mode: y)", pad=30)
    ax2.set_title("mean+std (array mode: xy)", pad=30)
    ax3.set_title("mean±std (table mode)", pad=30)
    ax4.set_title("aggregation by sns.lineplot()", pad=30)
    plt.suptitle("Input modes")
    plt.tight_layout()


def run(save=True, interactive=True, outdir="out/"):
    outdir = Path(outdir)
    sns.set(style="whitegrid")
    example_input1()
    save_figure(outdir/"example4/input_equivalence_1.png")
    example_input2()
    save_figure(outdir/"example4/input_equivalence_2.png")
    example_aggregation()
    save_figure(outdir/"example4/with_aggregation.png")
    if interactive:
        plt.show()


if __name__ == "__main__":
    run(save=True, interactive=True)
