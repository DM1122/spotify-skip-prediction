"""Helper functions for generating plots."""
# stdlib
import os

# external
import matplotlib
import matplotlib.pyplot as plt
from skopt import plots as skplot

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
