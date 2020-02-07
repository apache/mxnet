# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Common system utilities"""
from __future__ import absolute_import as _abs
import os
import tempfile
import shutil
try:
    import fcntl
except ImportError:
    fcntl = None


class TempDirectory(object):
    """Helper object to manage temp directory during testing.

    Automatically removes the directory when it went out of scope.
    """
    def __init__(self, custom_path=None):
        if custom_path:
            os.mkdir(custom_path)
            self.temp_dir = custom_path
        else:
            self.temp_dir = tempfile.mkdtemp()
        self._rmtree = shutil.rmtree

    def remove(self):
        """Remote the tmp dir"""
        if self.temp_dir:
            self._rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None

    def __del__(self):
        self.remove()

    def relpath(self, name):
        """Relative path in temp dir

        Parameters
        ----------
        name : str
            The name of the file.

        Returns
        -------
        path : str
            The concatenated path.
        """
        return os.path.join(self.temp_dir, name)

    def listdir(self):
        """List contents in the dir.

        Returns
        -------
        names : list
            The content of directory
        """
        return os.listdir(self.temp_dir)


def tempdir(custom_path=None):
    """Create temp dir which deletes the contents when exit.

    Parameters
    ----------
    custom_path : str, optional
        Manually specify the exact temp dir path

    Returns
    -------
    temp : TempDirectory
        The temp directory object
    """
    return TempDirectory(custom_path)


class FileLock(object):
    """File lock object

    Parameters
    ----------
    path : str
        The path to the lock
    """
    def __init__(self, path):
        self.lock_file = open(path, "w")
        if fcntl:
            fcntl.lockf(self.lock_file, fcntl.LOCK_EX)


    def release(self):
        """Release the lock"""
        if self.lock_file:
            if fcntl:
                fcntl.lockf(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None


def filelock(path):
    """Create a file lock which locks on path

    Parameters
    ----------
    path : str
        The path to the lock

    Returns
    -------
    lock : File lock object
    """
    return FileLock(path)


def is_source_path(path):
    """Check if path is source code path.

    Parameters
    ----------
    path : str
        A possible path

    Returns
    -------
    valid : bool
        Whether path is a possible source path
    """
    if os.path.exists(path):
        return True
    if path.find("\n") != -1:
        return False
    spath = path.rsplit(".", 1)
    return len(spath) == 2 and spath[1].strip() == spath[1]


def which(exec_name):
    """Try to find full path of exec_name

    Parameters
    ----------
    exec_name : str
        The executable name

    Returns
    -------
    path : str
        The full path of executable if found, otherwise returns None
    """
    base_list = ["", "/bin"] + os.environ.get("PATH", "").split(os.pathsep)
    for path in base_list:
        full_path = os.path.join(path, exec_name)
        if os.path.isfile(full_path) and os.access(full_path, os.X_OK):
            return full_path
    return None

def get_lower_ir(s):
    """Get lower ir code of a schedule.
    This is useful for debug, since you don't have to find all inputs/outputs
    for a schedule in a fused subgraph.

    Parameters
    ----------
    s: Schedule

    Returns
    -------
    ir: str
        The lower ir
    """
    from .. import tensor
    from ..build_module import lower

    outputs = s.outputs

    inputs = []
    def find_all(op):
        if isinstance(op, tensor.PlaceholderOp):
            inputs.append(op.output(0))
        else:
            for x in op.input_tensors:
                find_all(x.op)

    for out in outputs:
        find_all(out)

    return lower(s, inputs, simple_mode=True)
