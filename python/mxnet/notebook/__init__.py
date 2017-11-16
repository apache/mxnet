# pylint: disable=invalid-name, missing-docstring, no-init, old-style-class, multiple-statements

"""MXNet notebook: an easy to use visualization platform"""

try:
    import bokeh
except ImportError:
    class Bokeh_Failed_To_Import: pass
    bokeh = Bokeh_Failed_To_Import

try:
    import boken.io
except ImportError:
    pass
