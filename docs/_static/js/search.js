$(document).ready(function () {
    var searchForm = $("#search-input-wrap").children("form").first();
    searchForm.append('<div class="form-group searchBtn"><input type="submit" class="form-control" value="Go"></div>');
    searchForm.children("div").first().addClass("searchBox");
    searchForm.children("div").first().children("input").first().attr("placeholder", "press to search")
});