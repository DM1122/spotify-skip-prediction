"""Helper functions for generating plots."""
# stdlib
import logging
import os

# external
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skopt import plots as skplot

LOG = logging.getLogger(__name__)
matplotlib.use("TkAgg")


def plot_skopt_convergence(opt_res, path):
    """Plots the convergence plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure()
    skplot.plot_convergence(opt_res, ax=None, true_minumum=None, yscale=None)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "convergence")
    fig.show()


def plot_skopt_evaluations(opt_res, path):
    """Plots the evaluations plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    skplot.plot_evaluations(result=opt_res, bins=32, dimensions=None, plot_dims=None)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "evaluations")
    fig.show()


def plot_skopt_objective(opt_res, path):
    """Plots the objective plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    skplot.plot_objective(
        result=opt_res,
        levels=64,
        n_points=64,
        n_samples=256,
        size=2,
        zscale="linear",
        dimensions=None,
        sample_source="random",
        minimum="result",
        n_minimum_search=None,
        plot_dims=None,
        show_points=True,
        cmap="viridis_r",
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "objective")
    fig.show()


def plot_skopt_regret(opt_res, path):
    """Plots the regret plot from the skopt package.

    Args:
        opt_res (scipy.optimize.OptimizeResult): Optimization result object.
        path (str): Directory at which to save plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    fig = plt.figure()
    skplot.plot_regret(opt_res, ax=None, true_minumum=None, yscale=None)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path + "regret")
    fig.show()


def plot_series(
    df,
    title,
    traces,
    indicies=None,
    x_errors=None,
    y_errors=None,
    title_x="x",
    title_y="y",
    size=None,
    draw_mode="lines+markers",
    vlines=None,
    hlines=None,
    dark_mode=False,
):
    """Plot series.

    Args:
        df (pandas.core.DataFrame): Pandas dataframe
        title (str): Plot title
        traces (list[str]): Column names of traces to graph
        indicies (list[str], optional): Column names of indicies for trace to use.
            Defaults to None.
        x_errors (str, optional): Column name of x-errors. Defaults to None.
        y_errors ([type], optional): Column name of y-errors. Defaults to None.
        title_x (str, optional): X-axis title. Defaults to "x".
        title_y (str, optional): Y-axis title. Defaults to "y".
        size (tuple, optional): Width and height of plot in pixels. Defaults to None.
        draw_mode (str, optional): Whether to render lines or markers or both.
            Defaults to "lines+markers".
        vlines (list[int], optional): List of ordinates for vertical lines.
            Defaults to None.
        hlines (list[int], optional): List of ordinates for horizontal lines.
            Defaults to None.
        dark_mode (bool, optional): Dark mode. Defaults to False.
    """
    fig = go.Figure()

    template = "plotly" if not dark_mode else "plotly_dark"

    # draw traces
    for i, trace in enumerate(traces):
        name = trace
        y = df[trace]
        index = (
            df[indicies[i]] if indicies is not None else df.index
        )  # cannot have one trace use df.index and others not. WIP
        x_error = df[x_errors[i]] if x_errors is not None else None
        y_error = df[y_errors[i]] if y_errors is not None else None

        # error bars
        x_error = {
            "type": "data",
            "symmetric": True,
            "array": x_error,
            # "color":"black",
            # "thickness":1.5,
            # "width":3
        }

        y_error = {
            "type": "data",
            "symmetric": True,
            "array": y_error,
            # "color":"black",
            # "thickness":1.5,
            # "width":3
        }

        fig.add_trace(
            go.Scatter(
                arg=None,
                cliponaxis=None,
                connectgaps=False,
                customdata=None,
                customdatasrc=None,
                dx=None,
                dy=None,
                error_x=x_error,
                error_y=y_error,
                fill=None,
                fillcolor=None,
                groupnorm=None,
                hoverinfo=None,
                hoverinfosrc=None,
                hoverlabel=None,
                hoveron=None,
                hovertemplate=None,
                hovertemplatesrc=None,
                hovertext=None,
                hovertextsrc=None,
                ids=None,
                idssrc=None,
                legendgroup=None,
                line=None,
                marker=None,
                meta=None,
                metasrc=None,
                mode=draw_mode,
                name=name,
                opacity=None,
                orientation=None,
                r=None,
                rsrc=None,
                selected=None,
                selectedpoints=None,
                showlegend=True,
                stackgaps=None,
                stackgroup=None,
                stream=None,
                t=None,
                text=None,
                textfont=None,
                textposition=None,
                textpositionsrc=None,
                textsrc=None,
                texttemplate=None,
                texttemplatesrc=None,
                tsrc=None,
                uid=None,
                uirevision=None,
                unselected=None,
                visible=None,
                x=index,
                x0=None,
                xaxis=None,
                xcalendar=None,
                xsrc=None,
                y=y,
                y0=None,
                yaxis=None,
                ycalendar=None,
                ysrc=None,
            )
        )

    # draw vertical lines
    if vlines is not None:
        for vline in vlines:
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=vline,
                y0=0,
                x1=vline,
                y1=1,
                line={"color": "black", "width": 2, "dash": "dash"},
            )

    # draw horizontal lines
    if hlines is not None:
        for hline in hlines:
            fig.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                y0=hline,
                x1=1,
                y1=hline,
                line={"color": "black", "width": 2, "dash": "dash"},
            )

    # config plot
    w = size[0] if size is not None else None
    h = size[1] if size is not None else None

    fig.update_layout(
        title=title,
        width=w,
        height=h,
        xaxis_title=title_x,
        yaxis_title=title_y,
        template=template,
    )

    fig.show()
