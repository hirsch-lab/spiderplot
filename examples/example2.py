"""
Examples demonstrating the support for different data formats:
    - long-form,
    - wide-form, and
    - arrays
"""

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from spiderplot import spiderplot, spiderplot_facet
from commons import save_figure


def get_sample_data():
    vals1 = [["sneezing", 3.1, 4.1, 7.1],
             ["laughing", 1.5, 7.4, 7.8],
             ["drinking", 1.1, 5.5, None],
             ["shopping", None, 3.7, 5.3],
             ["learning", 6.4, 2.8, 5.1],
             ["sporting", 7.8, 4.3, 1.8]]
    df1 = pd.DataFrame(vals1, columns=["activity","20y-40y","40y-60y","60y-80y"])
    df1 = pd.melt(df1, id_vars=["activity"], var_name="age", value_name="level")

    vals2 = [["sneezing", 2.9, 3.2, 7.6],
             ["laughing", 2.8, 7.3, 7.9],
             ["drinking", 0.7, 5.1, 8.5],
             ["shopping", 3.3, 3.5, 5.0],
             ["learning", 5.5, 2.9, 4.4],
             ["sporting", 7.7, 4.7, 1.3]]
    df2 = pd.DataFrame(vals2, columns=["activity","20y-40y","40y-60y","60y-80y"])
    df2 = pd.melt(df2, id_vars=["activity"], var_name="age", value_name="level")
    df = pd.concat([df1, df2], keys=["male","female"])
    df.index.names = ["sex", None]
    df = df.reset_index(level=0)
    df = df.reset_index(drop=True)
    return df


def example_array():
    """
    Use spiderplot(x, y, **kwargs) like a normal matplotlib function.
    """
    df = get_sample_data()
    df = df[df["sex"]=="male"]
    df_young = df[df["age"] == "20y-40y"]
    df_med   = df[df["age"] == "40y-60y"]
    df_old   = df[df["age"] == "60y-80y"]

    plt.figure()
    spiderplot(x=df_young["activity"], y=df_young["level"],
               label="young", color="blue")
    spiderplot(x=df_med["activity"], y=df_med["level"],
               label="medium", color="green")
    spiderplot(x=df_old["activity"], y=df_old["level"],
               label="old", color="red")
    plt.gca().set_axisbelow(True)
    plt.legend(loc="upper right",
               bbox_to_anchor=(1.4, 1.1),
               borderaxespad=0.)
    plt.tight_layout()


def example_long():
    plt.figure()
    df = get_sample_data()
    spiderplot(x="activity", y="level", hue="age", style="sex", data=df,
               fill=True, fillalpha=0.3, mec="#222222", alpha=0.8, lw=1.5)
    plt.legend(loc="upper right",
               bbox_to_anchor=(1.4, 1.1),
               borderaxespad=0.)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()


def example_facet():
    df = get_sample_data()
    spiderplot_facet(data=df, col="age", hue="sex",
                     x="activity", y="level",
                     #rlabel="", legendtitle="Tataa!",
                     fill=True, fillalpha=0.3, offset=0., direction=1,
                     sharex=False, sharey=False)
    plt.legend(loc="upper right",
               bbox_to_anchor=(1.5, 1.1),
               borderaxespad=0.)
    plt.tight_layout()


def run(save=True, interactive=True, outdir="out/"):
    outdir = Path(outdir)
    sns.set(style="whitegrid")
    example_array()
    save_figure(outdir/"example2/sample_array.png", enabled=save)
    example_long()
    save_figure(outdir/"example2/sample_long_form.png", enabled=save)
    example_facet()
    save_figure(outdir/"example2/sample_facet.png", enabled=save)
    if interactive:
        plt.show()


if __name__ == "__main__":
    run(save=True, interactive=True)
