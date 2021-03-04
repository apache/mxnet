'use strict';

$(window).on('load', function() {
    $('#side-nav').resizable( "destroy" );
    $('#nav-tree-contents > ul > li > div.item').remove();
    $('#nav-tree-contents > ul > li > ul.children_ul').css({'display': 'block'});
    $('#doc-content').css('margin-left', '300px');
    $('#nav-btn').on('click', function() { $('#side-nav').css('display', 'flex') });
});