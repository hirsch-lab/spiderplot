import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def _ensure_axes(ax, enforce):
    if ax is None:
        if enforce:
            ax = plt.subplot(projection="polar")
        else:
            ax = plt.gca()
    if isinstance(ax, mpl.axes.Axes) and ax.name != "polar":
        msg = ("Axes must use polar projection. Use one of the following "
               "statements to ensure polar projection:\n"
               "    plt.gca(polar=True)\n"
               "    ax = plt.subplot(..., polar=True)\n"
               "    fig, ax = plt.subplots(subplot_kw={'polar': True})\n"
               )
        raise ValueError(msg)
    return ax


def _format_input_data(x, y, hue, style, data):
    fmt = "invalid"
    if (y is None and data is None):
        raise ValueError("Arguments y and data cannot both be None.")
    if data is None:
        if y is None:
            msg = "In array mode (data=None), argument y must be set."
            raise ValueError(msg)
        if x is None:
            x = pd.RangeIndex(len(y))
        else:
            x = pd.Index(x)
        data = pd.DataFrame(y).set_index(x)
        data.index.name = "x"
        data.columns.name = "category"
        # For pandas > 1.1:
        # data = data.melt(ignore_index=False, value_name="value").reset_index()
        data = data.reset_index().melt(value_name="value", id_vars="x")
        x = "x"
        y = "value"
        hue = "category"
        fmt = "long"
    elif isinstance(data, pd.Series):
        data.name = "value" if data.name is None else data.name
        name = data.name
        data = data.to_frame().reset_index()
        x = "index"
        y = name
        fmt = "long"
    elif isinstance(data, pd.DataFrame):
        data = data.copy()
        if x is None and y is None:
            fmt = "wide"
        else:
            fmt = "long"

    assert(fmt != "invalid")
    return fmt, x, y, hue, style, data


def _adjust_polar_grid(ax, vals, labels, offset, direction, color):
    ax.set_thetagrids(vals/np.pi*180, labels, color=color)
    ax.set_theta_offset(np.pi/2+offset)
    ax.set_theta_direction(direction)
    ax.set_rlabel_position(0)
    ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment="center")
    ax.tick_params(axis="y", which="both", labelsize=8)
    ax.set_xlabel(None)
    ax.set_ylabel(None)


def _compute_theta(x, y, data):
    if data is not None:
        # If x is col name: get column, else: its a vector.
        x = data.get(x,x)
        y = data.get(y,y)
    if x is None:
        x = list(range(len(y if y is not None else data)))
    x = pd.Series(x)
    x_vals = x.unique()
    n_axes = len(x_vals)
    t_vals = np.linspace(0, 2*np.pi, n_axes, endpoint=False)
    theta = x.map(dict(zip(x_vals, t_vals)))
    return theta, t_vals, x_vals


def _fill_and_close(ax, data, extent, lines_old,
                    fill, fillalpha, fillcolor, kwargs):
    # This is the fragile part and modify/add the artists.
    # - Close the lines
    # - Create a polygon patch if fill=True

    # If kwargs["label"] exists, the new line artist will carry that label.
    # Line2D.get_label() might be None, though it should always be a string.
    has_label, label = "label" in kwargs, kwargs.get("label", None)
    lines_new = [(l.get_label(),l) for l in ax.lines if
                 id(l) not in lines_old and
                 ((has_label and l.get_label()==label) or
                  (not has_label and l.get_label().startswith("_line")))]

    from matplotlib.patches import Polygon
    patches = []
    for _,l in lines_new:
        xy = l.get_xydata()
        # Filter nan-valued items. This step is necessary for seaborn>=0.10.
        xy = xy[~np.isnan(xy).any(axis=1)]

        if fill:
            # Add patches.
            alpha = fillalpha if fillalpha is not None else l.get_alpha()
            color = fillcolor if fillcolor is not None else l.get_color()
            xyp = xy.copy()

            if extent is not None:
                if data is not None:
                    extent = data.get(extent, extent)
                extent = np.asarray(extent)
                xyp = np.concatenate([xyp,[xyp[0]]], axis=0)
                extent = np.append(extent, extent[0])
                xy1 = xyp.copy()
                xy1[:,1] += extent
                xy2 = xyp.copy()
                xy2[:,1] -= extent
                xyp = np.concatenate([xy1,xy2[::-1]], axis=0)

            poly = Polygon(xyp, closed=True, fc=color, alpha=alpha)
            poly.set_fc(color)
            ax.add_patch(poly)
        # Close lines.
        if True:
            xdata, ydata = l.get_xdata(), l.get_ydata()
            # pandas>=0.10 doesn't skip nan-points, leading to interrupted
            # lines (with the segments that use a nan-point missing). I
            # consider this the right behavior. If the nan-filtered xy
            # should be used:
            #xdata, ydata = xy.T
            if len(xdata):
                l.set_xdata(np.concatenate([xdata,[xdata[0]]]))
            if len(ydata):
                l.set_ydata(np.concatenate([ydata,[ydata[0]]]))


def spiderplot(x=None, y=None, hue=None, size=None,
               style=None, extent=None, data=None,
               fill=True, fillalpha=0.25, fillcolor=None,
               offset=0., direction=-1,
               ax=None, _enforce_polar=True, **kwargs):
    """
    Create a spider chart with x defining the axes and y the values.

    The function is based on seaborn's lineplot() using a polar projection.
    The parameters indicated by (*) are specific to spiderplot(). The o
    For a more detailed documentation of the function arguments, see:
    https://seaborn.pydata.org/generated/seaborn.lineplot.html

    Arguments:
        x, y:           Vectors if data is None, else column keys of data.
        hue:            Vector or key in data. Grouping variable that will
                        produce lines with different colors.
        size:           Vector or key in data. Grouping variable that will
                        produce lines with different widths.
        style:          Vector or key in data. Grouping variable that will
                        produce lines with different dashes and/or markers.
        extent (*):     Vector or constant or key in data. Variable with
                        the error information per data point. Use this to
                        indicate error bounds: y±error
        data:           pandas.DataFrame or None. Data in long- or wide form.
                        Can be None if the data is provided through x and y.
        fill (*):       Fill area. Default: enabled
        fillalpha (*):  Alpha value for fill polygon. Default: 0.25
        fillcolor (*):  Color for fill polygon. Default: None (automatic)
        offset (*):     Offset of the polar plot in degree.
        direction (*):  Either -1 or +1. Plot CW:-1 or CCW:+1. Default: -1.
        ax:             Pre-existing axes for the plot, if available.
        **kwargs:       Additional arguments will be forwarded to
                        sns.lineplot()

    """
    DEFAULTS = dict(markers=True,
                    markeredgecolor=None,
                    alpha=0.7)
    defaults = DEFAULTS.copy()
    defaults.update(kwargs)
    kwargs = defaults

    ax = _ensure_axes(ax=ax, enforce=_enforce_polar)
    ret = _format_input_data(x=x, y=y, hue=hue, style=style, data=data)
    fmt, x, y, hue, style, data = ret

    theta, t_vals, x_vals = _compute_theta(x, y, data)

    # Keep track of newly added lines.
    lines_old = {id(l) for l in ax.lines}

    # Create line plot.
    # Note: this is similar to data.plot.area(), but uses seaborn
    # semantics. See also this feature request for areaplot():
    # https://github.com/mwaskom/seaborn/issues/2410

    if fmt == "wide":
        index_to_theta = dict(zip(data.index.values, theta))
        pos_to_label = dict(zip(range(len(theta)), data.index.values))
        data.index = data.index.map(index_to_theta)
        ax = sns.lineplot(data=data, ax=ax, **kwargs)

    elif fmt == "long":
        ax = sns.lineplot(x=theta, y=y, hue=hue, size=size, style=style,
                          data=data, ax=ax, **kwargs)

    _fill_and_close(ax=ax,
                    data=data,
                    extent=extent,
                    lines_old=lines_old,
                    fill=fill,
                    fillalpha=fillalpha,
                    fillcolor=fillcolor,
                    kwargs=kwargs)

    if _enforce_polar or False:
        _adjust_polar_grid(ax=ax, vals=t_vals, labels=x_vals,
                           offset=offset, direction=direction, color="gray")
    if fmt == "wide":
        ax.set_xticklabels(list(pos_to_label.values()))

    return ax


def spiderplot_facet(data, row=None, col=None, hue=None,
                     x=None, y=None, style=None,
                     rlabel=None, legendtitle=None,
                     fill=True, fillalpha=0.3, offset=0., direction=-1,
                     **kwargs):
    """

    Arguments:
        kwargs:     Are forwarded to sns.FacetGrid(**kwargs)

    See plotting_demo.py for a demonstration.
    """
    # Don't drop nans! This will completely mess up the diagram!
    grid = sns.FacetGrid(data=data, row=row, col=col, hue=hue, dropna=False,
                         subplot_kws=dict(projection="polar"), despine=False,
                         **kwargs)
    grid.map_dataframe(spiderplot, x=x, y=y, style=style,
                       fill=fill, fillalpha=fillalpha, offset=offset,
                       direction=direction, _enforce_polar=False)
    rlabel = rlabel if rlabel else y
    grid.set(xlabel=rlabel)
    grid.fig.subplots_adjust(wspace=.4, hspace=.4)
    if legendtitle:
        grid.add_legend(title=legendtitle)
    for ax in grid.axes.ravel():
        _, t_vals, x_vals = _compute_theta(x, y, data)
        _adjust_polar_grid(ax=ax,
                           vals=t_vals,
                           labels=x_vals,
                           offset=offset,
                           direction=direction,
                           color="gray")
    return grid
