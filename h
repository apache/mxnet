<!doctype html>
<!--[if IcaE 9]> <html class="ie9 no-js" lang="{{ shop.locale }}"> <![endif]-->
<!--[if (gt IE 9)|!(IE)]><!--> <html class="no-js" lang="{{ shop.locale }}"> <!--<![endif]-->
<head>
 
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
  <meta name="theme-color" content="{{ settings.color_button }}">
  <link rel="canonical" href="{{ canonical_url }}">

  {%- if settings.favicon != blank -%}
    <link rel="shortcut icon" href="{{ settings.favicon | img_url: '32x32' }}" type="image/png">
  {%- endif -%}

  {%- capture seo_title -%}
    {%- if template == 'search' and search.performed == true -%}
      {{ 'general.search.heading' | t: count: search.results_count }}: {{ 'general.search.results_with_count' | t: terms: search.terms, count: search.results_count }}
    {%- else -%}
      {{ page_title }}
    {%- endif -%}
    {%- if current_tags -%}
      {%- assign meta_tags = current_tags | join: ', ' -%} &ndash; {{ 'general.meta.tags' | t: tags: meta_tags -}}
    {%- endif -%}
    {%- if current_page != 1 -%}
      &ndash; {{ 'general.meta.page' | t: page: current_page }}
    {%- endif -%}
    {%- assign escaped_page_title = page_title | escape -%}
    {%- unless escaped_page_title contains shop.name -%}
      &ndash; {{ shop.name }}
    {%- endunless -%}
  {%- endcapture -%}
  <title>{{ seo_title | strip }}</title>

  {%- if page_description -%}
    <meta name="description" content="{{ page_description | escape }}">
  {%- endif -%}

  {% include 'social-meta-tags' %}

  {{ 'theme.scss.css' | asset_url | stylesheet_tag }}

  <script>
    var theme = {
      strings: {
        addToCart: {{ 'products.product.add_to_cart' | t | json }},
        soldOut: {{ 'products.product.sold_out' | t | json }},
        unavailable: {{ 'products.product.unavailable' | t | json }},
        regularPrice: {{ 'products.product.regular_price' | t | json }},
        sale: {{ 'products.product.on_sale' | t | json }},
        showMore: {{ 'general.filters.show_more' | t | json }},
        showLess: {{ 'general.filters.show_less' | t | json }},
        addressError: {{ 'sections.map.address_error' | t | json }},
        addressNoResults: {{ 'sections.map.address_no_results' | t | json }},
        addressQueryLimit: {{ 'sections.map.address_query_limit_html' | t | json }},
        authError: {{ 'sections.map.auth_error_html' | t | json }},
        newWindow: {{ 'general.accessibility.link_messages.new_window' | t | json }},
        external: {{ 'general.accessibility.link_messages.external' | t | json }},
        newWindowExternal: {{ 'general.accessibility.link_messages.new_window_and_external' | t | json }}
      },
      moneyFormat: {{ shop.money_format | json }}
    }

    document.documentElement.className = document.documentElement.className.replace('no-js', 'js');
  </script>

  <!--[if (lte IE 9) ]>{{ 'match-media.min.js' | asset_url | script_tag }}<![endif]-->

  {%- if template.directory == 'customers' -%}
    <!--[if (gt IE 9)|!(IE)]><!--><script src="{{ 'shopify_common.js' | shopify_asset_url }}" defer="defer"></script><!--<![endif]-->
    <!--[if lte IE 9]><script src="{{ 'shopify_common.js' | shopify_asset_url }}"></script><![endif]-->
  {%- endif -%}

  <!--[if (gt IE 9)|!(IE)]><!--><script src="{{ 'lazysizes.js' | asset_url }}" async="async"></script><!--<![endif]-->
  <!--[if lte IE 9]><script src="{{ 'lazysizes.min.js' | asset_url }}"></script><![endif]-->

  <!--[if (gt IE 9)|!(IE)]><!--><script src="{{ 'vendor.js' | asset_url }}" defer="defer"></script><!--<![endif]-->
  <!--[if lte IE 9]><script src="{{ 'vendor.js' | asset_url }}"></script><![endif]-->

  <!--[if (gt IE 9)|!(IE)]><!--><script src="{{ 'theme.js' | asset_url }}" defer="defer"></script><!--<![endif]-->
  <!--[if lte IE 9]><script src="{{ 'theme.js' | asset_url }}"></script><![endif]-->

  {{ content_for_header }}
{% include 'alireviews_core' %} 
 </head>

<a href='https://www.bonanza.com/booths/HarmonyM16?fref=vGBXjEO5'>Check out my booth, HarmonyM16</a>
<body class="template-{{ template | split: '.' | first }}">

  <a class="in-page-link visually-hidden skip-link" href="#MainContent">
  <div id="SearchDrawer" class="search-bar drawer drawer--top" role="dialog" aria-modal="true" aria-label="{{ 'general.search.placeholder' | t }}">
    <div class="search-bar__table">
      <div class="search-bar__table-cell search-bar__form-wrapper">
        <form class="search search-bar__form" action="/search" method="get" role="search">
          <input class="search__input search-bar__input" type="search" name="q" value="{{ search.terms | escape }}" placeholder="{{ 'general.search.placeholder' | t }}" aria-label="{{ 'general.search.placeholder' | t }}">
          <button class="search-bar__submit search__submit btn--link" type="submit">
            {% include 'icon-search' %}
            <span class="icon__fallback-text">{{ 'general.search.submit' | t }}</span>
          </button>
        </form>
      </div>
      <div class="search-bar__table-cell text-right">
        <button type="button" class="btn--link search-bar__close js-drawer-close">
          {% include 'icon-close' %}
          <span class="icon__fallback-text">{{ 'general.search.close' | t }}</span>
        </button>
      </div>
    </div>
  </div>

  {% section 'header' %}

  <div class="page-container" id="PageContainer">

    <main class="main-content js-focus-hidden" id="MainContent" role="main" tabindex="-1">
      {{ content_for_layout }}
    </main>
<iframe width="472" height="273" src="https://www.youtube.com/embed/N1G-4GjC-iY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    {% section 'footer' %}

    <div id="slideshow-info" class="visually-hidden" aria-hidden="true">
      {{- 'sections.slideshow.navigation_instructions' | t -}}
    </div>

  </div>

  <ul hidden>
    <li id="a11y-refresh-page-message">{{ 'general.accessibility.refresh_page' | t }}</li>
  </ul>
  </html>
{% include "fireapps-aliorder-bulk-action-edit-product" %}
