=======
表示例
=======

タイポグラフィ
==============

.. code-block:: rst

   ===========
   h1. Heading
   ===========

   h2. Heading
   ===========

   -----------
   h3. Heading
   -----------

   h4. Heading
   -----------

.. raw:: html

   <h1>h1. Heading</h1>
   <h2>h2. Heading</h2>
   <h3>h3. Heading</h3>
   <h4>h4. Heading</h4>
   <h5>h5. Heading</h5>
   <h6>h6. Heading</h6>

画像
====

.. code-block:: rst

   .. image:: img/sample.png

.. image:: img/sample.png

.. code-block:: rst

   .. figure:: img/sample.png
      :scale: 50 %
      :alt: sample image

.. figure:: img/sample.png
   :scale: 50 %
   :alt: sample image

テーブル
========

.. code-block:: rst

    +---------+-----------+------------+-------------------+
    |         | 使用例    |  書き方    |  HTMLタグ         |
    +=========+===========+============+===================+
    |強調     |*文字列*   | \*で囲む   | <em>              |
    +---------+-----------+------------+-------------------+
    |強い強調 |**文字列** | \*\*で囲む | <strong>          |
    +---------+-----------+------------+-------------------+
    |コード   |``文字列`` |\`\`で囲む  |<span class="pre"> |
    +---------+-----------+------------+-------------------+

+---------+-----------+------------+-------------------+
|         | 使用例    |  書き方    |  HTMLタグ         |
+=========+===========+============+===================+
|強調     |*文字列*   | \*で囲む   | <em>              |
+---------+-----------+------------+-------------------+
|強い強調 |**文字列** | \*\*で囲む | <strong>          |
+---------+-----------+------------+-------------------+
|コード   |``文字列`` |\`\`で囲む  |<span class="pre"> |
+---------+-----------+------------+-------------------+

コード
======

::

    ふつうの文章::

        コードブロック

    ふつうの文章



.. code-block:: rst

  .. code-block:: python

        import sys

        print sys.path

引用
====

.. code-block:: rst

       | これらの行は
       | ソースファイルの通りに
       | 改行されます。

**example**

   | これらの行は
   | ソースファイルの通りに
   | 改行されます。

ダウンロード用リンク
====================

**rst**

.. code-block:: rst

    :download:`this file <examples.rst>`

**出力例**

:download:`this file <./examples.rst>`


警告
====

**Hint**

.. code-block:: rst

    .. hint::

        This is a hint directive!

.. hint::

    This is a **hint** directive!

**Note**

.. code-block:: rst

    .. note::

        This is a note directive!

.. note::

    This is a **note** directive!

**Warning**

.. code-block:: rst

    .. warning::

        This is a warning directive!

.. warning::

    This is a **warning** directive!

**Tip**

.. code-block:: rst

    .. tip::

        This is a tip directive!

.. tip::

    This is a **tip** directive!


**Important**

.. code-block:: rst

    .. important::

        This is a important directive!

.. important::

    This is a **important** directive!

**Error**

.. code-block:: rst

    .. error::

        This is a error directive!

.. error::

    This is a **error** directive!

**Caution**

.. code-block:: rst

    .. caution::

        This is a caution directive!

.. caution::

    This is a caution directive!

**Danger**

.. code-block:: rst

    .. danger::

        This is a danger directive!

.. danger::

    This is a **danger** directive!

