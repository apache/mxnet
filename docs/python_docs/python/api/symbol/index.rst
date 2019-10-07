.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.

mxnet.symbol
============

The Symbol API in Apache MXNet is an interface for symbolic programming. It features the use of computational graphs, reduced memory usage, and pre-use function optimization.


Example
-------

The following example shows how you might build a simple expression with the Symbol API.

.. code-block:: python

	import mxnet as mx
	# Two placeholers are created with mx.sym.variable
	a = mx.sym.Variable('a')
	b = mx.sym.Variable('b')
	# The symbol is constructed using the '+' operator
	c = a + b
	(a, b, c)


Tutorials
---------
.. container:: cards

   .. card::
      :title: Symbol Guide
      :link: ../../tutorials/packages/symbol/

      The Symbol guide. Start here!

Symbol Package
--------------

.. container:: cards

   .. card::
      :title: Symbol
      :link: symbol.html

      Symbolic programming using the Symbol API.

.. toctree::
   :hidden:
   :maxdepth: 2
   :glob:

   symbol
   */index