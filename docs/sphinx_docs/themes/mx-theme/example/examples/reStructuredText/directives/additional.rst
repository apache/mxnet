========================
Additional body elements
========================

Table of Contents
=================

.. code-block:: rst

   .. contents:: Here's a very long Table of
      Contents title

.. contents:: Here's a very long Table of
   Contents title

Container
=========

.. code-block:: rst

    .. container:: custom

       This paragraph might be rendered in a custom way.

.. container:: custom

   This paragraph might be rendered in a custom way.

Topic
=====

.. code-block:: rst

    .. topic:: Topic Title

        Subsequent indented lines comprise
        the body of the topic, and are
        interpreted as body elements.

.. topic:: Topic Title

    Subsequent indented lines comprise
    the body of the topic, and are
    interpreted as body elements.

Epigraph
========

.. code-block:: rst

    .. epigraph::

        No matter where you go, there you are.

        -- Buckaroo Banzai

.. epigraph::

   No matter where you go, there you are.

   -- Buckaroo Banzai

Compound Paragraph
==================

.. code-block:: rst

    .. compound::

        The 'rm' command is very dangerous.  If you are logged
        in as root and enter ::

            cd /
            rm -rf *

        you will erase the entire contents of your file system.

.. compound::

   The 'rm' command is very dangerous.  If you are logged
   in as root and enter ::

       cd /
       rm -rf *

   you will erase the entire contents of your file system.
