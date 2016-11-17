/* Initial sidebar toc toggle button*/
$(document).ready(function () {
    var allEntry = $("div.sphinxsidebarwrapper li");
    allEntry.each(function () {
        $(this).prepend("<span class='tocToggle' onclick='toggle(this)'></span>");
        var childUL = $(this).find("ul");
        if(childUL.length && childUL.first().children().length) {
            $(this).addClass("closed");
            $(this).find("ul").first().hide();
        }
        else 
            $(this).addClass("leaf");
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