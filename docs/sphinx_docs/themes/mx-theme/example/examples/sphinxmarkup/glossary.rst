========
Glossary
========

.. code-block:: rst

    .. glossary::

        environment
            A structure where information about all documents under the root is
            saved, and used for cross-referencing.  The environment is pickled
            after the parsing stage, so that successive runs only need to read
            and parse new and changed documents.

        source directory
            The directory which, including its subdirectories, contains all
            source files for one Sphinx project.

.. glossary::

   environment
      A structure where information about all documents under the root is
      saved, and used for cross-referencing.  The environment is pickled
      after the parsing stage, so that successive runs only need to read
      and parse new and changed documents.

   source directory
      The directory which, including its subdirectories, contains all
      source files for one Sphinx project.


.. code-block:: rst

    .. glossary::

        term 1 : A
        term 2 : B
            Definition of both terms.

.. glossary::

   term 1 : A
   term 2 : B
      Definition of both terms.

