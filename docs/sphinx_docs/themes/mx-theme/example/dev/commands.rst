==============
Build Commands
==============

|project|\ 's build command list.

packaging
---------

Create a source distribution (tar.gz).

.. code-block:: bat

   python setup.py sdist

Upload PyPI
-------------

upload pckage to `PyPI <https://pypi.python.org/pypi>`_.

.. code-block:: bat

   python setup.py register sdist upload

Build Example's Document
------------------------

Generate HTML Document for Example.

.. code-block:: bat

   sphinx-build -b html ./example ./_build -c ./example
