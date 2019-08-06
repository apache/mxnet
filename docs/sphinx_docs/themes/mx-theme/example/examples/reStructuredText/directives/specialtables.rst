==============
Special tables
==============

Table
=====

.. code-block:: rst

    .. table:: Truth table for "not"
        :widths: auto

        =====  =====
            A    not A
        =====  =====
        False  True
        True   False
        =====  =====

.. table:: Truth table for "not"
   :widths: auto

   =====  =====
     A    not A
   =====  =====
   False  True
   True   False
   =====  =====

CSV Table
=========

.. code-block:: rst

    .. csv-table:: Frozen Delights!
        :header: "Treat", "Quantity", "Description"
        :widths: 15, 10, 30

        "Albatross", 2.99, "On a stick!"
        "Crunchy Frog", 1.49, "If we took the bones out, it wouldn't be
        crunchy, now would it?"
        "Gannet Ripple", 1.99, "On a stick!"

.. csv-table:: Frozen Delights!
   :header: "Treat", "Quantity", "Description"
   :widths: 15, 10, 30

   "Albatross", 2.99, "On a stick!"
   "Crunchy Frog", 1.49, "If we took the bones out, it wouldn't be
   crunchy, now would it?"
   "Gannet Ripple", 1.99, "On a stick!"

List Table
==========

.. code-block:: rst

    .. list-table:: Frozen Delights!
        :widths: 15 10 30
        :header-rows: 1

        * - Treat
            - Quantity
            - Description
        * - Albatross
            - 2.99
            - On a stick!
        * - Crunchy Frog
            - 1.49
            - If we took the bones out, it wouldn't be
            crunchy, now would it?
        * - Gannet Ripple
            - 1.99
            - On a stick!

.. list-table:: Frozen Delights!
   :widths: 15 10 30
   :header-rows: 1

   * - Treat
     - Quantity
     - Description
   * - Albatross
     - 2.99
     - On a stick!
   * - Crunchy Frog
     - 1.49
     - If we took the bones out, it wouldn't be
       crunchy, now would it?
   * - Gannet Ripple
     - 1.99
     - On a stick!
