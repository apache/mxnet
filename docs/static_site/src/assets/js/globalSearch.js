$(document).ready(function () {
    docsearch({
        apiKey: '500f8e78748bd043cc6e4ac130e8c0e7',
        indexName: 'apache_mxnet',
        inputSelector: '#global-search',
        algoliaOptions: { 'facetFilters': ["version:master"] },
        debug: true// Set debug to true if you want to inspect the dropdown
    });
    
    $("#search-icon").hover(function () {
        $(".trigger").hide();
        $("#global-search-form").css("display", "inline-block");
        $("#global-search").animate({
            width: "300px"
        }).focus();
    });

    $("#global-search").blur(function () {
        $(this).animate({
            width: "0px"
        }, function () {
            $("#global-search-form").hide();
            $(".trigger").show();
        });
    });
});