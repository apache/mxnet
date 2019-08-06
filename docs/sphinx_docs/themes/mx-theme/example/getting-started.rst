================
Getting Started
================

Instllation
===========

------------------
Installing Sphinx
------------------


.. code-block:: shell

   pip install -U Sphinx

-----------------------------------------
Installing Sphinx Material Design Theme.
-----------------------------------------

.. code-block:: shell

   pip install sphinx_materialdesign_theme


Usage
========

Add the following line to :code:`conf.py`.

.. code-block:: python

   html_theme = 'sphinx_materialdesign_theme'

   # Html logo in drawer.
   # Fit in the drawer at the width of image is 240 px.
   html_logo = '_static/logo.jpg'


Html theme options
==================

-------------------------
Customize menus in header
-------------------------

``header_links`` option is used to specify a list of menu in secondary heaer row.

.. code-block:: rst

   'header_links' : [
        ('Home', 'index', False, 'home'),
        ("ExternalLink", "http://example.com", True, 'launch'),
        ("NoIconLink", "http://example.com", True, ''),
        ("GitHub", "https://github.com/myyasuda/sphinx_materialdesign_theme", True, 'link')
    ]


--------------------
Customize css colors
--------------------

.. code-block:: rst

       'primary_color': 'indigo',
       'accent_color': 'pink',

Let's try to select color.

**Primary Colors**

.. raw:: html

   <style type="text/css">
       .color-pick-container {
          display: flex;
          flex-wrap: wrap;
          padding: 0 0 20px 0;
       }
       .color-pick-container > button {
           margin: 0 5px 5px 0;
           color: white !important;
           width: 125px;
           height: 48px;
       }
       .color-pick-container > button[name="indigo"] {
           background-color: rgb(63,81,181) !important;
       }
       .color-pick-container > button[name="blue"] {
           background-color: rgb(33,150,243) !important;
       }
       .color-pick-container > button[name="light_blue"] {
           background-color: rgb(3,169,244) !important;
       }
       .color-pick-container > button[name="cyan"] {
           background-color: rgb(0,188,212) !important;
       }
       .color-pick-container > button[name="teal"] {
           background-color: rgb(0, 150, 136) !important;
       }
       .color-pick-container > button[name="green"] {
           background-color: rgb(76, 175, 80) !important;
       }
       .color-pick-container > button[name="light_green"] {
           background-color: rgb(139, 195, 74) !important;
       }
       .color-pick-container > button[name="lime"] {
           background-color: rgb(205, 220, 57) !important;
       }
       .color-pick-container > button[name="yellow"] {
           background-color: rgb(255, 235, 59) !important;
       }
       .color-pick-container > button[name="amber"] {
           background-color: rgb(255, 193, 7) !important;
       }
       .color-pick-container > button[name="orange"] {
           background-color: rgb(255, 152, 0) !important;
       }
       .color-pick-container > button[name="brown"] {
           background-color: rgb(121, 85, 72) !important;
       }
       .color-pick-container > button[name="blue_grey"] {
           background-color: rgb(96, 125, 139) !important;
       }
       .color-pick-container > button[name="grey"] {
           background-color: rgb(158, 158, 158) !important;
       }
       .color-pick-container > button[name="deep_orange"] {
           background-color: rgb(255, 87, 34) !important;
       }
       .color-pick-container > button[name="red"] {
           background-color: rgb(244, 67, 54) !important;
       }
       .color-pick-container > button[name="pink"] {
           background-color: rgb(233, 30, 99) !important;
       }
       .color-pick-container > button[name="purple"] {
           background-color: rgb(156, 39, 176) !important;
       }
       .color-pick-container > button[name="deep_purple"] {
           background-color: rgb(103, 58, 183) !important;
       }
   </style>

   <div class="color-pick-container">
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="indigo">indigo</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="blue">blue</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="light_blue">light blue</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="cyan">cyan</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="teal">teal</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="green">green</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="light_green">light green</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="lime">lime</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="yellow">yellow</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="amber">amber</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="orange">orange</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="brown">brown</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="blue_grey">blue grey</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="grey">grey</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="deep_orange">deep orange</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="red">red</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="pink">pink</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="purple">purple</button>
        <button class="primary-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="deep_purple">deep purple</button>
   </div>

**Accent Colors**

.. raw:: html

   <div class="color-pick-container">
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="indigo">indigo</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="blue">blue</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="light_blue">light blue</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="cyan">cyan</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="teal">teal</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="green">green</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="light_green">light green</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="lime">lime</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="yellow">yellow</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="amber">amber</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="orange">orange</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="deep_orange">deep orange</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="red">red</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="pink">pink</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="purple">purple</button>
        <button class="accent-color-pick-button mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" name="deep_purple">deep purple</button>
    </div>
   <script>
   $(function(){
        var $primaryColor = $('.primary-color-pick-button');
        var $accentColor = $('.accent-color-pick-button');
        var primary = 'indigo';
        var accent = 'pink';
        function toggle() {
            $primaryColor.each(function(index, e) {
                var $e = $(e);
                var name = $e.attr('name');
                if (name === accent) {
                    $e.prop("disabled", true);
                } else {
                    $e.prop("disabled", false);
                }
            });
            $accentColor.each(function(index, e) {
                var $e = $(e);
                var name = $e.attr('name');
                if (name === primary) {
                    $e.prop("disabled", true);
                } else {
                    $e.prop("disabled", false);
                }
            });
        }
        function changeCss(){
            var $css = $('link[href*="material-design-lite"]');
            var href = $css.attr('href');
            $css.attr('href', href.replace(/material\..*-.*\.min\.css/g, 'material.' + primary + '-' + accent + '.min.css'))
        }

        $primaryColor.click(function() {
            var $this = $(this);
            primary = $this.attr('name');
            toggle();
            changeCss();
        });
        $accentColor.click(function() {
            var $this = $(this);
            accent = $this.attr('name');
            toggle();
            changeCss();
        });

        toggle();
   });
   </script>


---------------
Conf.py example
---------------

The following is a description of the options that can be specified in ``html_theme_options`` in your project's ``conf.py``.

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
