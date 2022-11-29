import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import warnings

class SpiderWarning(Warning): pass



def _ensure_axes(ax, enforce):
    if ax is None:
        # Get axes without creating new one.
        fig = plt.gcf()
        if fig.axes:
            ax = plt.gca()
    if isinstance(ax, mpl.axes.Axes) and type(ax).__name__.startswith("Polar"):
        return ax
    else:
        if enforce:
            ax = plt.subplot(polar=True)
            return ax
        else:
            msg = ("Axes must use polar projection. Use one of the following "
                   "statements to ensure polar projection:\n"
                   "    ax = plt.subplot(..., polar=True)\n"
                   "    fig, ax = plt.subplots(subplot_kw={'polar': True})\n"
                   )
            raise ValueError(msg)


def _format_input_data(x, y, hue, style, data):
    fmt = "invalid"
    if (y is None and data is None):
        raise ValueError("Arguments y and data cannot both be None.")
    if data is None:
        # Array mode: only x (optionally) and y are provided.
        if y is None:
            msg = "In array mode (data=None), argument y must be set."
            raise ValueError(msg)
        if x is None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                x = y.index.copy()
            else:
                x = pd.RangeIndex(len(y))
        else:
            x = pd.Index(x)
        data = pd.DataFrame(y).set_index(x)
        data.index.name = "_x"
        data.columns.name = "category"
        data = data.reset_index().melt(value_name="_value", id_vars="_x")
        x = "_x"
        y = "_value"
        # The case where y.shape[1]=d>1 can be resolved in two ways:
        #   1) hue="category": Each column represents a separate category,
        #      represented as differently colored lines in the plot
        #   2) hue=None: Each row represents d samples of the same category.
        #      Before plotting, the data is aggregated along axis=1. A single
        #      line is plotted by sns.lineplot ± uncertainty bounds.
        # Problem: Choice 2) is easy by setting hue=None. However, choice 1)
        # cannot be made by the user, because hue should be "category", which
        # is an arbitrarily chosen, internal variable name.
        # Resolution: introduce magic setting hue="columns", which applies
        # only to this array mode. I think this is most consistent with
        # seaborn.
        if isinstance(hue, str) and hue=="columns":
            hue = "category"
        fmt = "long"
    elif isinstance(data, pd.Series):
        data = data.copy()
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


def _compute_theta(x, y, data,
                   n_ticks_hint=None,
                   is_categorical=True,
                   is_closed=False):
    """
    Args:
        x, y, data:     See spiderplot()
        n_ticks_hint:   Number of ticks on the theta-axis.
        is_categorical: Switch between categorical and numeric mode.
        is_closed:      If start and end map to same point on (cyclic)
                        theta-axis. Is ignored if is_categorical=False.
    """
    if data is not None:
        # If x is col name: get column, else: its a vector.
        x = data.get(x,x)
        y = data.get(y,y)
    if x is None:
        x = list(range(len(y if y is not None else data)))
    if is_categorical:
        x = pd.Series(x)
        x_vals = x.unique()
        n_vals = len(x_vals)
        t_vals = np.linspace(0, 2*np.pi, n_vals, endpoint=is_closed)
        theta = x.map(dict(zip(x_vals, t_vals)))
        if n_ticks_hint is not None:
            step = int(max(np.round(len(x_vals)/n_ticks_hint), 1))
            x_vals = x_vals[::step]
            t_vals = t_vals[::step]
    else:
        x_min = x.min()
        x_max = x.max()
        theta = (x-x.min())/(x.max()-x.min())*2*np.pi
        if n_ticks_hint is None:
            n_ticks_hint = 8
        t_vals = np.linspace(0, 2*np.pi, n_ticks_hint, endpoint=False)
        x_vals = np.linspace(x.min(), x.max(), n_ticks_hint, endpoint=False)
    return theta, t_vals, x_vals


def _adjust_polar_grid(ax, vals, labels,
                       offset, direction,
                       color):
    ax.set_theta_offset(np.pi/2+offset)
    ax.set_theta_direction(direction)
    ax.set_thetagrids(vals/np.pi*180, labels, color=color)
    ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment="center")
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", which="both", labelsize=8)
    ax.set_xlabel(None)
    ax.set_ylabel(None)


def _fill_and_close(ax, data, extent, lines_old,
                    fill, fillalpha, fillcolor, kwargs):
    """
    This is the fragile part. Modify/add the artists:
    - Close the lines
    - Create a polygon patch if fill=True
    """
    def _draw_poly(xy, color, alpha, ax):
        poly = Polygon(xy, closed=True, fc=color, alpha=alpha)
        poly.set_fc(color)
        ax.add_patch(poly)

    def _draw_extent(xy, extent, data, color, alpha, ax):
        if data is not None:
            extent = data.get(extent, extent)
        extent = np.asarray(extent)
        xy = np.concatenate([xy, [xy[0]]], axis=0)
        extent = np.append(extent, extent[0])
        xy1 = xy.copy()
        xy1[:,1] += extent
        xy2 = xy.copy()
        xy2[:,1] -= extent
        xy = np.concatenate([xy1, xy2[::-1]], axis=0)
        _draw_poly(xy=xy, color=color, alpha=alpha, ax=ax)

    draw_fill = fill
    draw_extent = extent is not None
    close_lines = True

    # Select all lines that have been added by sns.lineplot(). Unfortunately,
    # seaborn does not return any handles to lines it has created. Thus,
    # we need to reverse-engineer part of the logic of sns.lineplot()...
    #
    # WARNING: If ever the below code needs fixing, check out the examples
    # of lineplot and see how seaborn adds lines to plots:
    #   https://seaborn.pydata.org/generated/seaborn.lineplot.html
    #
    #       flights = sns.load_dataset("flights")
    #       ax = sns.lineplot(data=flights, x="year", y="passengers")
    #       plt.show()
    #       print([ (l.get_label(),l, len(l.get_xydata())) for l in ax.lines ])
    #
    # If kwargs["label"] exists, the new line artist will carry that label.
    # Line2D.get_label() might be None, though it should always be a string.
    #
    # Historical note: In earlier versions, seaborn created lines with labels
    # that started with "_line". Current versions use labels starting with
    # "_child". As this is internal logic, it is better to avoid checking
    # for the labels, except they were set explicitly.
    has_label, label = "label" in kwargs, kwargs.get("label", None)
    lines_new = [(l.get_label(),l) for l in ax.lines if
                 id(l) not in lines_old and
                 ((has_label and l.get_label()==label) or
                  (not has_label and len(l.get_xdata())>0))]

    if len(lines_new) == 0:
        msg = ("Failed to retrieve the lines of the spiderplot. It is "
               "likely that the aesthetics of the resulting plot are "
               "dissatisfactory. Please report the issue: "
               "https://github.com/hirsch-lab/spiderplot")
        warnings.warn(msg, SpiderWarning, stacklevel=2)

    patches = []
    for _,l in lines_new:
        # Filter nan-valued items. This step is necessary for seaborn>=0.10.
        xy = l.get_xydata()
        mask = np.isnan(xy).any(axis=1)
        xy = xy[~mask]
        xyp = xy.copy()

        alpha = fillalpha if fillalpha is not None else l.get_alpha()
        color = fillcolor if fillcolor is not None else l.get_color()

        if draw_extent:
            if mask.any():
                msg = "Extent is skipped for line containing nan data."
                warnings.warn(msg, SpiderWarning)
            else:
                _draw_extent(xy=xyp, extent=extent, data=data,
                             color=color, alpha=alpha, ax=ax)
        if draw_fill:
            _draw_poly(xy=xyp, color=color, alpha=alpha, ax=ax)

        if close_lines:
            xdata, ydata = l.get_xdata(), l.get_ydata()
            # pandas>=0.10 doesn't skip nan-points, leading to interrupted
            # lines (with the segments that use a nan-point missing). I
            # consider this the right behavior. If the nan-filtered xy
            # should be used:
            #xdata, ydata = xy.T
            if len(xdata):
                l.set_xdata(np.concatenate([xdata,[xdata[0]]]))
            if len(ydata):
                l.set_ydata(np.concatenate([ydata,[ydata[0]]]))

    # Readjust axes limits to see the patches.
    if draw_extent:
        ax.relim()
        ax.autoscale_view()


def spiderplot(x=None, y=None, hue=None, size=None,
               style=None, extent=None, data=None,
               fill=True, fillalpha=0.2, fillcolor=None,
               rref=None, rref_kws=None,
               offset=0., direction=-1,
               n_ticks_hint=None,
               is_categorical=True,
               ax=None, _enforce_polar=True, **kwargs):
    """
    Create a spider chart with x defining the axes and y the values.

    The function is based on seaborn's lineplot() using a polar projection.
    The parameters indicated by (*) are specific to spiderplot().
    For a more detailed documentation of the function arguments, see:
    https://seaborn.pydata.org/generated/seaborn.lineplot.html

    Similar to seaborn functions, spiderplot accepts different data formats:
        - array mode:       x, y and other parameters are np.arrays
        - long-form mode:   x, y, hue, size, extent can be keys of data
        - wide-form mode:   only argument data is required, where
                            x: row indices; and y: table values (data.values),
                            with the columns representing different categories.

    spiderplot() makes sense most for categorical x-data, even though
    numerical data can also be passed. See argument is_categorical.

    Args:
        x, y:           Vectors if data is None, else column keys of data.
        hue:            Vector or key in data. Grouping variable that will
                        produce lines with different colors.
                        In array mode (data=None, x and y data arrays),
                        hue="columns" will plot lines for each column; if
                        hue=None, data aggregation is enabled.
        size:           Vector or key in data. Grouping variable that will
                        produce lines with different widths.
        style:          Vector or key in data. Grouping variable that will
                        produce lines with different dashes and/or markers.
        extent:     (*) Vector or constant or key in data. Variable with
                        the error information per data point. Use this to
                        indicate error bounds: y±error
        data:           pandas.DataFrame or None. Data in long- or wide form.
                        Can be None if the data is provided through x and y.
        fill:       (*) Fill area. Default: enabled
        fillalpha:  (*) Alpha value for fill polygon. Default: 0.25
        fillcolor:  (*) Color for fill polygon. Default: None (automatic)
        rref,       (*) Highlight the iso-line at value rref. The keywords
        rref_kws:       rref_kws can be used to adjust the appearance of
                        this reference line.
        offset:     (*) Offset of the polar plot in degrees.
        direction:  (*) Either -1 or +1. Plot CW:-1 or CCW:+1. Default: -1.
        n_ticks_hint:   Number of ticks along the x-axis. By default,
                    (*) spiderplot() uses all values for categorical data,
                        and n=8 for numerical data.
        is_categorical: Switch between categorical and numerical mode.
                    (*) Determines how the x-data is interpreted and how the
                        tick-locations are computed.
        ax:             Pre-existing axes for the plot, if available.
        **kwargs:       Additional arguments will be forwarded to
                        sns.lineplot().

    Returns:
        ax:             The matplotlib axes containing the plot.

    """
    # Override defaults from matplotlib: markeredgecolor is white by default,
    # leading to a halo around the marker. Matplotlib's property aliases
    # (mec for markeredgecolor) makes this a bit more complicated.
    DEFAULTS = dict(markers=True,
                    markeredgecolor=None,
                    alpha=0.7)
    ALIASES = dict(mec="markeredgecolor")
    for key, key_new in ALIASES.items():
        kwargs[key_new] = kwargs.pop(key, DEFAULTS[key_new])
    defaults = DEFAULTS.copy()
    defaults.update(kwargs)
    kwargs = defaults

    ax = _ensure_axes(ax=ax, enforce=_enforce_polar)
    ret = _format_input_data(x=x, y=y, hue=hue, style=style, data=data)
    fmt, x, y, hue, style, data = ret

    theta, t_vals, x_vals = _compute_theta(x=x, y=y, data=data,
                                           n_ticks_hint=n_ticks_hint,
                                           is_categorical=is_categorical)
    # Keep track of newly added lines.
    lines_old = {id(l) for l in ax.lines}

    # Create line plot.
    # Note: this is similar to data.plot.area(), but uses seaborn
    # semantics. See also this feature request for areaplot():
    # https://github.com/mwaskom/seaborn/issues/2410

    if fmt == "wide":
        index_to_theta = dict(zip(data.index.values, theta))
        pos_to_label = dict(zip(range(len(theta)), data.index.values))
        data.index = data.index.map(lambda x: index_to_theta[x])
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
                           offset=offset, direction=direction,
                           color="gray")
    if fmt == "wide":
        ax.set_xticklabels(list(pos_to_label.values()))

    if rref is not None:
        rref_kws = {"color":"k", "lw": 0.5} if rref_kws is None else rref_kws
        t = np.linspace(0, 2*np.pi, 100)
        ax.plot(t, np.ones_like(t)*rref, **rref_kws)
    return ax


def spiderplot_facet(data, row=None, col=None, hue=None,
                     x=None, y=None, style=None,
                     sharex=False, sharey=False,
                     fill=True, fillalpha=0.2,
                     rref=None, rref_kws=None,
                     offset=0., direction=-1,
                     n_ticks_hint=None,
                     is_categorical=True,
                     **kwargs):
    """
    Create an sns.FacetGrid using spiderplot().

    The function is based on seaborn's FacetGrid(). For more details
    see: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html

    The use of sns.FacetGrid in combination with spiderplot() is a bit tricky.
    In particular, dropping NaNs mess up the diagram.

    Args:
        data:           pandas.DataFrame or None. Data in long- or wide form.
                        Can be None if the data is provided through x and y.
        row, col, hue:  Keys in data. Variables that define subsets of the
                        data, which will be drawn on separate facets in the
                        grid.
        sharex, sharey: Shares axes across figure. Disabled by default.
        **kwargs:       Additional keyword arguments are forwarded to
                        sns.FacetPlot().

        x, y,
        style,
        fill,
        fillalpha,
        fillcolor,
        rref,
        rref_kws,
        offset,
        direction,
        ax:             Same as in spiderplot()
    """
    # Don't drop nans! This will completely mess up the diagram!
    grid = sns.FacetGrid(data=data, row=row, col=col, hue=hue, dropna=False,
                         subplot_kws=dict(projection="polar"), despine=False,
                         sharex=sharex, sharey=sharey,
                         **kwargs)
    grid.map_dataframe(spiderplot, x=x, y=y, style=style,
                       fill=fill, fillalpha=fillalpha,
                       rref=rref, rref_kws=rref_kws,
                       offset=offset, direction=direction,
                       _enforce_polar=False)
    grid.fig.subplots_adjust(wspace=.4, hspace=.4)
    for ax in grid.axes.ravel():
        _, t_vals, x_vals = _compute_theta(x, y, data,
                                           is_categorical=is_categorical,
                                           n_ticks_hint=n_ticks_hint)
        _adjust_polar_grid(ax=ax,
                           vals=t_vals,
                           labels=x_vals,
                           offset=offset,
                           direction=direction,
                           color="gray")
    return grid
