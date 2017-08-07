"""A module is like a FeedForward model. But we would like to make it
easier to compose, similar to Torch modules.
"""

from .base_module import BaseModule
from .module import Module
from .bucketing_module import BucketingModule
from .sequential_module import SequentialModule

from .python_module import PythonModule, PythonLossModule
