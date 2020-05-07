$(document).ready(function () {
    const globalSearch = docsearch({
        apiKey: '500f8e78748bd043cc6e4ac130e8c0e7',
        indexName: 'apache_mxnet',
        inputSelector: '#global-search',
        algoliaOptions: { 'facetFilters': ["version:1.6"], 'hitsPerPage': 5 },
        debug: false// Set debug to true if you want to inspect the dropdown
    });

    $("#search-icon").click(function () {
        $(".trigger").fadeOut("fast", function () {
            $("#global-search-form").css("display", "inline-block");
            $("#global-search-close").show();
            $("#global-search-dropdown-container").show();
            $("#global-search").animate({
                width: "300px"
            }).focus();
        });
    });

    $("#global-search-close").click(function () {
        $("#global-search-dropdown-container").hide();
        $("#global-search").animate({
            width: "0px"
        }, function () {
            $(this).hide();
            $("#global-search-form").hide();
            $(".trigger").fadeIn("fast");
        });
    });

    $("#global-search-dropdown-container").click(function (e) {
        $(".gs-version-dropdown").toggle();
        e.stopPropagation();
    });

    $("ul.gs-version-dropdown li").each(function () {
        $(this).on("click", function () {
            $("#global-search").val("");
            $("li.gs-opt.active").removeClass("active");
            $(this).addClass("active");
            $("#gs-current-version-label").html(this.innerHTML);
            globalSearch.algoliaOptions = {
                'facetFilters': ["version:" + this.innerHTML], 'hitsPerPage': 5
            };
        });
    });

    $(document).click(function () {
        $(".gs-version-dropdown").hide();
    });

});