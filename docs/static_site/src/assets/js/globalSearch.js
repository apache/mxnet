$(document).ready(function () {
    docsearch({
        apiKey: '500f8e78748bd043cc6e4ac130e8c0e7',
        indexName: 'apache_mxnet',
        inputSelector: '#global-search',
        algoliaOptions: { 'facetFilters': ["version:master"] },
        debug: true// Set debug to true if you want to inspect the dropdown
    });
    
    $("#search-icon").click(function () {
        $(".trigger").fadeOut("fast", function() {
            $("#global-search-form").css("display", "inline-block");
            $("#global-search-close").show();
            $("#global-search").animate({
                width: "300px"
            }).focus();
        });
    });

    $("#global-search-close").click(function () {
        $("#global-search").animate({
            width: "0px"
        }, function () {
            $(this).hide();
            $("#global-search-form").hide();
            $(".trigger").fadeIn("fast");
        });
    });
});