var searchBox = $("#search-input-wrap");
var APIsubMenu;
$("#burgerMenu").children().each(function () {
    if($(this).children().first().html() == 'API') APIsubMenu = $(this).clone()
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
        if(plusMenuList[i].html().length > 20) {
            $("#plusMenu").append(APIsubMenu);
        }
        else {
            $("#plusMenu").append("<li></li>");
            plusMenuList[i].removeClass("main-nav-link");
            $("#plusMenu").children().last().append(plusMenuList[i]);
        }
    }
};


$(document).ready(function () {
    navbar();
    $(window).resize(function () {
        navbar();
    });
});