"""Utility functions."""
import os
import sys

def mkdirs_p(path):
    """Make directory recursively if not exists.

    Parameters
    ----------
    path : str
        The destination directory to be created.
    """
    if sys.version_info[0:2] >= (3, 2):
        os.makedirs(path, exist_ok=True)
    else:
        try:
            os.makedirs(path)
        except:
            pass
    assert os.path.isdir(path), "Unable to create directory: {}".format(path)
