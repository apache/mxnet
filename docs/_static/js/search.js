$(document).ready(function () {
    var searchForm = $("#search-input-wrap").children("form").first();
    searchForm.append('<div class="form-group searchBtn"><input type="submit" class="form-control" value="Go"></div>');
    searchForm.children("div").first().addClass("searchBox");
    $(".searchBox").addClass("searchBoxNorm");
    
    $('#searchIcon').click(function () {
        if($('#search-input-wrap').is(':hidden')) {
            $('#search-input-wrap').show();
            $('#searchIcon span').removeClass('glyphicon-search');
            $('#searchIcon span').addClass('glyphicon-remove-circle');
        }
        else {
            $('#search-input-wrap').hide();
            $('#searchIcon span').removeClass('glyphicon-remove-circle');
            $('#searchIcon span').addClass('glyphicon-search');
        }
    });
});