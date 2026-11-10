"""
Matplotlib figure helpers.

Why this file exists
--------------------
Matplotlib keeps a reference to every figure built through pyplot in a global
registry. Streamlit re-runs the whole script whenever a widget changes, so a
figure that is never closed stays in memory until the process exits. Over a
working session that shows up as memory use that only ever climbs, and on a
hosted server it ends with the app being killed.

How to use it
-------------
Draw figures with show_figure() instead of st.pyplot(). It draws the figure and
releases it in the same step:

    from figure_utils import show_figure

    fig, ax = plt.subplots()
    ax.plot(x, y)
    show_figure(fig)

Call close_all_figures() once at the start of a run to drop anything an earlier
run left behind.

Closing a figure does not make it unusable. The object stays valid, so code that
keeps a figure for a PDF report or a download button can still call savefig on
it afterwards. Only the pyplot registry entry goes away.
"""

from __future__ import annotations

import matplotlib

# The app renders plots to an image buffer and never opens a window, so use the
# non-interactive backend. This keeps GUI toolkits from being loaded, which
# matters most in the packaged desktop builds.
try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = [
    "show_figure",
    "close_figure",
    "close_all_figures",
    "open_figure_count",
]

# Streamlit still accepts use_container_width but prints a deprecation notice on
# every call. Translating it here keeps the call sites unchanged and the logs quiet.
_WIDTH_FOR_FLAG = {True: "stretch", False: "content"}


def _as_figure(obj):
    """
    Return the Figure behind obj, or None if there is none.

    Accepts a Figure, or a Seaborn grid such as the value returned by
    sns.clustermap, which holds its Figure in .figure (and .fig).
    """
    if isinstance(obj, Figure):
        return obj

    for name in ("figure", "fig"):
        candidate = getattr(obj, name, None)
        if isinstance(candidate, Figure):
            return candidate

    return None


def _normalise_width(kwargs: dict) -> dict:
    """Convert the retired use_container_width flag into the width argument."""
    flag = kwargs.pop("use_container_width", None)

    if flag is not None and "width" not in kwargs:
        width = _WIDTH_FOR_FLAG.get(bool(flag))
        if width is not None:
            kwargs["width"] = width

    return kwargs


def show_figure(obj, **kwargs):
    """
    Draw a figure in the app, then release it.

    Takes the same keyword arguments as st.pyplot. Pass a Figure or a Seaborn
    grid. The figure is closed once it has been drawn, so it stops taking up
    memory, but the object itself remains usable for savefig.
    """
    figure = _as_figure(obj)
    kwargs = _normalise_width(kwargs)

    try:
        st.pyplot(figure if figure is not None else obj, **kwargs)
    finally:
        close_figure(figure)


def close_figure(obj):
    """Release a single figure. Accepts a Figure, a Seaborn grid, or None."""
    figure = _as_figure(obj) if obj is not None else None

    if figure is None:
        return

    try:
        plt.close(figure)
    except Exception:
        pass


def close_all_figures():
    """
    Release every figure pyplot is still holding.

    Safe to call at the start of a run. A figure that has already been drawn, or
    that is being kept for a report, can still be written to a file afterwards.
    """
    try:
        plt.close("all")
    except Exception:
        pass


def open_figure_count() -> int:
    """
    How many figures pyplot is holding right now.

    Useful when checking whether figures are being released. Returns -1 if the
    count cannot be read.
    """
    try:
        from matplotlib import _pylab_helpers

        return len(_pylab_helpers.Gcf.figs)
    except Exception:
        return -1
