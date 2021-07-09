#!/usr/bin/env python3

import sys

from flake8.main import git

if __name__ == '__main__':
    sys.exit(
        git.hook(
            # (optional):
            # any value > 0 enables complexity checking with mccabe
            complexity=0,

            # (optional):
            # if True, this returns the total number of error which will cause
            # the hook to fail
            strict=True,

            # (optional):
            # a comma-separated list of errors and warnings to ignore
            # ignore=

            # (optional):
            # allows for the instances where you don't add the the files to the
            # index before running a commit, e.g git commit -a
            lazy=git.config_for('lazy'),
        )
    )
