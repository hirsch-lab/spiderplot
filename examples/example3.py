"""
Examples demonstrating the support for different data formats:
    - long-form,
    - wide-form, and
    - arrays
"""

import string
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from spiderplot import spiderplot, spiderplot_facet
from commons import save_figure


def format_data(df, fmt):
    if fmt == "wide":
        # Wide-form data-frame
        df = df.pivot(index="Month", columns="City", values="Temp_high")
        # pivot() sorts lexicographically. Reorder the index
        df = df.reindex(pd.to_datetime(df.index, format="%b")
                        .sort_values().strftime("%b"))
        df.columns.name = None
        return df
    elif fmt == "long":
        # Long-form data frame.
        df = df.melt(id_vars=["City", "Month"],
                     value_name="Value",
                     var_name="Variable")
        return df
    elif fmt == "arrays":
        df = format_data(df=df, fmt="wide")
        x = df.index.values
        y = df.values
        cities = (np.ones_like(y)*range(df.shape[1])).flatten("F")
        cities = df.columns.values[cities.astype(int)]
        return x, y, cities
    else:
        assert(False)


def get_sample_data(fmt="long", cities=None, add_nans=True):
    filepath = Path(__file__).parent / "climate_data.csv"
    df = pd.read_csv(filepath, usecols=["City", "Month", "Temp_high"])
    if cities is not None:
        df = df[df["City"].isin(cities)]
    if add_nans:
        df = df.set_index(["City", "Month"])
        df.loc[("Zurich", "Jun")] = np.nan
        df.loc[("Zurich", "Sep")] = np.nan
        df.loc[("Shanghai", "Jan")] = np.nan
        df = df.reset_index()
    return format_data(df=df, fmt=fmt)


def example_long():
    df = get_sample_data(fmt="long")
    fig, ax = plt.subplots(num="Example: Long-format data frame",
                           subplot_kw={"polar": True})
    spiderplot(x="Month", y="Value", hue="City", style="City", data=df,
               fill=True, fillalpha=0.2, palette="deep", dashes=False,
               lw=1.5, markers=["o","*","^","v"], ax=ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), borderaxespad=0.)
    ax.set_axisbelow(True)
    ax.set_title("Average daily max. temperature")
    plt.tight_layout()


def example_wide():
    df = get_sample_data(fmt="wide")
    fig, ax = plt.subplots(num="Example: Wide-format data frame",
                           subplot_kw={"polar": True})
    spiderplot(data=df, fill=True, fillalpha=0.2, palette="deep",
               dashes=False, lw=1.5, markers=["o","*","^","v"], ax=ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), borderaxespad=0.)
    ax.set_axisbelow(True)
    ax.set_title("Average daily max. temperature")
    plt.tight_layout()


def example_arrays():
    x, y, categories = get_sample_data(fmt="arrays")
    fig, ax = plt.subplots(num="Example: arrays",
                           subplot_kw={"polar": True})

    spiderplot(x=x,                 # shape: (n,)
               y=y,                 # shape: (n,d)
               hue=categories,      # is ignored
               style=categories,    # must be 1D: (n*d)
               fill=True,
               fillalpha=0.2,
               palette="deep",
               dashes=False,
               lw=1.5,
               markers=["o","*","^","v"],
               ax=ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), borderaxespad=0.)
    ax.set_axisbelow(True)
    ax.set_title("Average daily max. temperature")
    plt.tight_layout()


def example_facet_long():
    df = get_sample_data(fmt="long")
    grid = spiderplot_facet(data=df, col="City", hue="City",
                            x="Month", y="Value", style="City",
                            fill=True, fillalpha=0.2,
                            n_ticks_hint=None,
                            is_categorical=True,
                            col_wrap=2)
    grid.set_titles("City: {col_name}")
    grid.add_legend(title="Cities",
                    loc="upper left",
                    bbox_to_anchor=(0.9, 0.995))
    grid.fig.subplots_adjust(wspace=.5, hspace=.5)


def run(save=True, interactive=True, outdir="out/"):
    outdir = Path(outdir)
    sns.set(style="whitegrid")
    example_arrays()
    save_figure(outdir/"example3/climate_array.png")
    example_wide()
    save_figure(outdir/"example3/climate_wide_form.png")
    example_long()
    save_figure(outdir/"example3/climate_long_form.png")
    example_facet_long()
    save_figure(outdir/"example3/climate_facet_long.png")
    if interactive:
        plt.show()


if __name__ == "__main__":
    run(save=True, interactive=True)
