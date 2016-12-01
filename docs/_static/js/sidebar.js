/* Initial sidebar toc toggle button*/
$(document).ready(function () {
    var allEntry = $("div.sphinxsidebarwrapper li");
    var subEntry = $("div.sphinxsidebarwrapper").children("ul").first().children("li");
    if(subEntry.length == 1) {
        allEntry = subEntry.find("li");
        subEntry.children("a").hide();
        subEntry.children("ul").css("padding-left", "0");
        subEntry.parent().css("margin-left", "-20px");
    }
    allEntry.each(function () {
        $(this).prepend("<span class='tocToggle' onclick='toggle(this)'></span>");
        var childUL = $(this).find("ul");
        if(childUL.length && childUL.first().children().length) {
            $(this).addClass("closed");
            $(this).find("ul").first().hide();
        }
        else 
            $(this).addClass("leaf");
        var anchor = $(this).children("a").first();
        anchor.click(function () {
            autoExpand(anchor);
        });
    });
    $('body').show();
});

/*Sidebar toc toggle button behavior*/
function toggle(elem) {
    if($(elem).parent().hasClass("closed")) {
        $(elem).parent().find("ul").first().show();
        $(elem).parent().removeClass("closed").addClass("opened");
    }
    else if($(elem).parent().hasClass("opened")) {
        $(elem).parent().find("ul").first().hide();
        $(elem).parent().removeClass("opened").addClass("closed");
    }
}

/*Automatically expand child level while cilcking an entry*/
function autoExpand(elem) {
    elem.parent().removeClass("closed").addClass("opened");
    elem.parent().children("ul").first().show();
}