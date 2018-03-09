var searchBox = $("#search-input-wrap");
var TITLE = ['/install/', '/gluon/' , '/api/', '/docs/', '/github/', '/community/', ];
var DOC_TITLE = ['/faq/', '/tutorials/', '/architecture/', '/model_zoo/'];
var APISubmenu, versionSubmenu, docSubmenu;
$("#burgerMenu").children().each(function () {
    if($(this).children().first().html() == 'API') APISubmenu = $(this).clone();
    if($(this).children().first().html().startsWith('Versions')) versionSubmenu = $(this).clone();
    if($(this).children().first().html() == 'Docs') docSubmenu= $(this).clone();
});

function navbar() {
    var leftOffset = 40;
    var plusMenuList = [];
    var plusIconLeft =$("#search-input-wrap").offset().left - leftOffset;
    var isCovered = false;
    $("#main-nav").children().each(function () {
        var rightPos;
        if($(this).is(':hidden')) {
            $(this).show();
            rightPos = $(this).offset().left + $(this).width();
            $(this).hide;
        }
        else rightPos = $(this).offset().left + $(this).width();
        
        if(isCovered) {
            plusMenuList.push($(this).clone());
            $(this).hide();
        }
        else if(rightPos > plusIconLeft) {
            isCovered = true;
            $(".plusIcon").first().show();
            plusMenuList.push($(this).clone());
            $(this).hide();
        }
        else $(this).show();
    });
    
    if(plusMenuList.length == 0) {
        $(".plusIcon").first().hide();
        return;
    }
    $("#plusMenu").empty();
    for (var i = 0; i < plusMenuList.length; ++i) {
        if(plusMenuList[i].attr('id') == 'dropdown-menu-position-anchor') {
            $("#plusMenu").append(APISubmenu);
        }
        else if(plusMenuList[i].attr('id') == 'dropdown-menu-position-anchor-version') {
            $("#plusMenu").append(versionSubmenu);
        }
        else if(plusMenuList[i].attr('id') == 'dropdown-menu-position-anchor-docs') {
            $("#plusMenu").append(docSubmenu);
        }
        else {
            $("#plusMenu").append("<li></li>");
            plusMenuList[i].removeClass("main-nav-link");
            $("#plusMenu").children().last().append(plusMenuList[i]);
        }
    }
};

/*Show bottom border of current tab*/
function showTab() {
    var url = window.location.href;
    if(url.indexOf('/get_started/why_mxnet') != -1) return;
    for(var i = 0; i < TITLE.length; ++i) {
        if(url.indexOf(TITLE[i]) != -1) {
            var tab = $($('#main-nav').children().eq(i));
            if(!tab.is('a')) tab = tab.find('a').first();
            tab.css('border-bottom', '3px solid');
            return;
        }
    }
     for(var i = 0; i < DOC_TITLE.length; ++i) {
        if(url.indexOf(DOC_TITLE[i]) != -1) {
            var tab = $($('#main-nav').children().eq(3));
            if(!tab.is('a')) tab = tab.find('a').first();
            tab.css('border-bottom', '3px solid');
        }
     }
}

$(document).ready(function () {
    navbar();
    showTab();
    $(window).resize(function () {
        navbar();
        if($("body").prop("clientWidth") < 1000 || $('div.sphinxsidebar').css('visibility') == 'hidden') $('div.content').css('width', '100%');
        else $('div.content').css('width', 'calc(100% - 300px)');
    });
});