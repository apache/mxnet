================
セットアップ
================

インストール方法
================

下記のコマンドを実行して、テーマをインストールします。

.. code-block:: bat

   pip install sphinx_materialdesign_theme


conf.pyの設定
=============

.. code-block:: python

   import sphinx_materialdesign_theme

   html_theme = 'sphinx_materialdesign_theme'
   html_theme_path = [sphinx_materialdesign_theme.get_path()]

   # Html logo in navbar.
   # Fit in the navbar at the height of image is 37 px.
   html_logo = '_static/logo.jpg'


HTMLテーマオプション
====================

The following is a description of the options that can be specified in **html_theme_options** in conf.py.

.. code-block:: python

   html_theme_options = {
       # Specify a list of menu in Header.
       # Tuples forms:
       #  ('Name', 'external url or path of pages in the document', boolean, 'icon name')
       #
       # Third argument:
       # True indicates an external link.
       # False indicates path of pages in the document.
       #
       # Fourth argument:
       # Specify the icon name.
       # For details see link.
       # https://material.io/icons/
       'header_links' : [
           ('Home', 'index', False, 'home'),
           ("ExternalLink", "http://example.com", True, 'launch'),
           ("NoIconLink", "http://example.com", True, ''),
           ("GitHub", "https://github.com/myyasuda/sphinx_materialdesign_theme", True, 'link')
       ],

       # Customize css colors.
       # For details see link.
       # https://getmdl.io/customize/index.html
       #
       # Values: amber, blue, brown, cyan deep_orange, deep_purple, green, grey, indigo, light_blue,
       #         light_green, lime, orange, pink, purple, red, teal, yellow(Default: indigo)
       'primary_color': 'indigo',
       # Values: Same as primary_color. (Default: pink)
       'accent_color': 'pink',

       # Customize layout.
       # For details see link.
       # https://getmdl.io/components/index.html#layout-section
       'fixed_drawer': True,
       'fixed_header': True,
       'header_waterfall': True,
       'header_scroll': False,

       # Render title in header.
       # Values: True, False (Default: False)
       'show_header_title': False,
       # Render title in drawer.
       # Values: True, False (Default: True)
       'show_drawer_title': True,
       # Render footer.
       # Values: True, False (Default: True)
       'show_footer': True
   }
