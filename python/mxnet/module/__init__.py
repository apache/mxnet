"""A module is like a FeedForward model. but we would like to make it
easier to be composed. So it is more like the Torch modules.
"""

from .base_module import BaseModule
from .module import Module
from .bucketing_module import BucketingModule
from .sequential_module import SequentialModule

from .python_module import PythonModule, PythonLossModule
