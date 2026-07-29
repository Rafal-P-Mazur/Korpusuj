"""Plotting utilities for the desktop interface.

Heavy Matplotlib and NetworkX objects are loaded lazily to keep application startup responsive.
"""

_plot_stack = None


def get_plot_stack():
    global _plot_stack
    if _plot_stack is None:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        # Potrzebujemy obu backendów - jednego do interfejsu (Tezaurus), drugiego do plików (Wykresy)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.cm as cm
        import networkx as nx

        _plot_stack = {
            "plt": plt,
            "Figure": Figure,
            "FigureCanvasTkAgg": FigureCanvasTkAgg,
            "FigureCanvasAgg": FigureCanvasAgg, # Odzyskany silnik!
            "cm": cm,
            "nx": nx
        }
    return _plot_stack
