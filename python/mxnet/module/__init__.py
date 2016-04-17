"""A module is like a FeedForward model. but we would like to make it
easier to be composed. So it is more like the Torch modules.
"""

from .base_module import BaseModule
from .module import Module
from .bucketing_module import BucketingModule
from .sequential_module import SequentialModule

<<<<<<< HEAD
from .python_module import PythonModule
=======
from .python_module import PythonModule, PythonLossModule
>>>>>>> d14cf84ed354866bedafc289c1078da05c3ed0e8
