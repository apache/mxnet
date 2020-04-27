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
"""Optimizer API of MXNet."""

from . import (optimizer, contrib, updater, utils, sgd,
               sgld, signum, dcasgd, nag, adagrad,
               adadelta, adam, adamax, nadam, ftrl,
               ftml, lars, lamb, rmsprop)
# pylint: disable=wildcard-import
from .optimizer import *

from .updater import *

from .utils import *

from .sgd import *

from .sgld import *

from .signum import *

from .dcasgd import *

from .nag import *

from .adagrad import *

from .adadelta import *

from .adam import *

from .adamax import *

from .nadam import *

from .ftrl import *

from .ftml import *

from .lars import *

from .lamb import *

from .rmsprop import *

__all__ = optimizer.__all__ + updater.__all__ + ['contrib'] + sgd.__all__ + sgld.__all__ \
          + signum.__all__ + dcasgd.__all__ + nag.__all__ + adagrad.__all__ + adadelta.__all__ \
          + adam.__all__ + adamax.__all__ + nadam.__all__ + ftrl.__all__ + ftml.__all__ \
          + lars.__all__ + lamb.__all__ + rmsprop.__all__
