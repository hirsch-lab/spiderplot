import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ._spider import spiderplot

def format_data(mode, x, y):
    if mode=="arrays":
        return x, y
    else:
        d = y.shape[1]
        cols = ["case %02d"%i for i in range(d)]
        df = pd.DataFrame(y, index=x, columns=cols)
        df.index.name = "x"
        df.columns.name = "dataset"

        if mode=="long-form":
            return df.reset_index().melt(id_vars="x",
                                         value_name="value")
        if mode=="wide-form":
            return df

    assert False


def generate_data_pair(mode, n=12):
    rng = np.random.RandomState(42)
    x = ["%02dh"%i for i in range(n)]
    y = np.r_[[rng.uniform(0.1, 1, len(x)),
               rng.uniform(0.1, 0.2, len(x))]]
    sign = (2*(np.arange(len(x))%2))-1
    y = (y*sign).T
    return format_data(mode, x, y)


def generate_data(mode, n=12, d=2):
    rng = np.random.RandomState(42)
    x = ["%02dh"%i for i in range(n)]
    y = np.r_[[rng.uniform(-1, 1, len(x)) for i in range(d)]].T
    return format_data(mode, x, y)


def demo_single():
    df = generate_data(mode="long-form", d=1)
    sns.set_style("whitegrid")
    ax = spiderplot(x="x", y="value", hue="dataset", legend=False,
                    data=df, palette="husl", rref=0)
    ax.set_rlim([-1.4,1.4])
    plt.tight_layout()
    plt.show()


def demo_pair():
    df = generate_data_pair(mode="long-form")
    sns.set_style("whitegrid")
    ax = spiderplot(x="x", y="value", hue="dataset", style="dataset",
                    data=df, dashes=False, palette="husl", rref=0)
    ax.set_rlim([-1.4,1.4])
    ax.legend(loc="upper right",
               bbox_to_anchor=(1.4, 1.),
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def demo_multi():
    df = generate_data(mode="long-form", d=5)
    ax = spiderplot(x="x", y="value", hue="dataset",
                    data=df, palette="husl", rref=0)
    ax.set_rlim([-1.4,1.4])
    ax.legend(loc="upper right",
               bbox_to_anchor=(1.4, 1.),
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()


def demo_aggregate():
    df = generate_data(mode="long-form", n=24, d=10)
    means = df.groupby("x")["value"].mean()
    stds = df.groupby("x")["value"].std()
    sns.set_style("whitegrid")
    # ax = spiderplot(x="x", y="value", hue="dataset", style="dataset", data=df,
    #                 fill=False, markers=False, dashes=False, legend=False,
    #                 palette=["gray" for i in range(10)], alpha=0.3)
    ax = spiderplot(y=means, extent=stds, color="red", fillcolor="gray",
                    fill=False, rref=0, label="mean ± std")
    ax.set_rlim([-1.4,1.4])
    ax.legend(loc="upper right",
               bbox_to_anchor=(1.4, 1.),
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()
