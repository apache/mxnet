===========================
Lists and Quote-like blocks
===========================

Bullet Lists
============

.. code-block:: rst

    - This is the first bullet list item.  The blank line above the
    first list item is required; blank lines between list items
    (such as below this paragraph) are optional.

    - This is the first paragraph in the second item in the list.

    This is the second paragraph in the second item in the list.
    The blank line above this paragraph is required.  The left edge
    of this paragraph lines up with the paragraph above, both
    indented relative to the bullet.

    - This is a sublist.  The bullet lines up with the left edge of
        the text blocks above.  A sublist is a new list so requires a
        blank line above and below.

    - This is the third item of the main list.

    This paragraph is not part of the list.

- This is the first bullet list item.  The blank line above the
  first list item is required; blank lines between list items
  (such as below this paragraph) are optional.

- This is the first paragraph in the second item in the list.

  This is the second paragraph in the second item in the list.
  The blank line above this paragraph is required.  The left edge
  of this paragraph lines up with the paragraph above, both
  indented relative to the bullet.

  - This is a sublist.  The bullet lines up with the left edge of
    the text blocks above.  A sublist is a new list so requires a
    blank line above and below.

- This is the third item of the main list.

This paragraph is not part of the list.

Enumerated Lists
================

.. code-block:: rst

    1. Item 1 initial text.

       a) Item 1a.
       b) Item 1b.

    2. a) Item 2a.
       b) Item 2b.

1. Item 1 initial text.

   a) Item 1a.
   b) Item 1b.

2. a) Item 2a.
   b) Item 2b.

Definition Lists
================

.. code-block:: rst

   term 1
       Definition 1.

   term 2
       Definition 2, paragraph 1.

       Definition 2, paragraph 2.

   term 3 : classifier
       Definition 3.

   term 4 : classifier one : classifier two
       Definition 4.

term 1
    Definition 1.

term 2
    Definition 2, paragraph 1.

    Definition 2, paragraph 2.

term 3 : classifier
    Definition 3.

term 4 : classifier one : classifier two
    Definition 4.

Field Lists
===========

.. code-block:: rst

   :Date: 2001-08-16
   :Version: 1
   :Authors: - Me
             - Myself
             - I
   :Indentation: Since the field marker may be quite long, the second
   and subsequent lines of the field body do not have to line up
   with the first line, but they must be indented relative to the
   field name marker, and they must line up with each other.
   :Parameter i: integer

:Date: 2001-08-16
:Version: 1
:Authors: - Me
          - Myself
          - I
:Indentation: Since the field marker may be quite long, the second
   and subsequent lines of the field body do not have to line up
   with the first line, but they must be indented relative to the
   field name marker, and they must line up with each other.
:Parameter i: integer


Option Lists
============

.. code-block:: rst

    -a         Output all.
    -b         Output both (this description is quite long).
    -c arg     Output just arg.
    --long     Output all day long.

    -p         This option has two paragraphs in the description.
            This is the first.

            This is the second.  Blank lines may be omitted between
            options (as above) or left in (as here and below).

    --very-long-option  A VMS-style option.  Note the adjustment for
                        the required two spaces.

    --an-even-longer-option
            The description can also start on the next line.

    -2, --two  This option has two variants.

    -f FILE, --file=FILE  These two options are synonyms; both have
                        arguments.

    /V         A VMS/DOS-style option.

-a         Output all.
-b         Output both (this description is
           quite long).
-c arg     Output just arg.
--long     Output all day long.

-p         This option has two paragraphs in the description.
           This is the first.

           This is the second.  Blank lines may be omitted between
           options (as above) or left in (as here and below).

--very-long-option  A VMS-style option.  Note the adjustment for
                    the required two spaces.

--an-even-longer-option
           The description can also start on the next line.

-2, --two  This option has two variants.

-f FILE, --file=FILE  These two options are synonyms; both have
                      arguments.

/V         A VMS/DOS-style option.

Quoted Literal Blocks
=====================

.. code-block:: rst

    >> Great idea!
    >
    > Why didn't I think of that?

    You just did!  ;-)

>> Great idea!
>
> Why didn't I think of that?

You just did!  ;-)

Line Blocks
===========

.. code-block:: rst

    | Lend us a couple of bob till Thursday.
    | I'm absolutely skint.
    | But I'm expecting a postal order and I can pay you back
    as soon as it comes.
    | Love, Ewan.

| Lend us a couple of bob till Thursday.
| I'm absolutely skint.
| But I'm expecting a postal order and I can pay you back
  as soon as it comes.
| Love, Ewan.

.. code-block:: rst

    Take it away, Eric the Orchestra Leader!

        | A one, two, a one two three four
        |
        | Half a bee, philosophically,
        |     must, *ipso facto*, half not be.
        | But half the bee has got to be,
        |     *vis a vis* its entity.  D'you see?
        |
        | But can a bee be said to be
        |     or not to be an entire bee,
        |         when half the bee is not a bee,
        |             due to some ancient injury?
        |
        | Singing...

Take it away, Eric the Orchestra Leader!

    | A one, two, a one two three four
    |
    | Half a bee, philosophically,
    |     must, *ipso facto*, half not be.
    | But half the bee has got to be,
    |     *vis a vis* its entity.  D'you see?
    |
    | But can a bee be said to be
    |     or not to be an entire bee,
    |         when half the bee is not a bee,
    |             due to some ancient injury?
    |
    | Singing...


Doctest Blocks
==============

.. code-block:: rst

    This is an ordinary paragraph.

    >>> print 'this is a Doctest block'
    this is a Doctest block

    The following is a literal block::

        >>> This is not recognized as a doctest block by
        reStructuredText.  It *will* be recognized by the doctest
        module, though!

This is an ordinary paragraph.

>>> print 'this is a Doctest block'
this is a Doctest block

The following is a literal block::

    >>> This is not recognized as a doctest block by
    reStructuredText.  It *will* be recognized by the doctest
    module, though!
