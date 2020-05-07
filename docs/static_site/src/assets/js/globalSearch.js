$(document).ready(function () {
    let globalSearch = docsearch({
        apiKey: '500f8e78748bd043cc6e4ac130e8c0e7',
        indexName: 'apache_mxnet',
        inputSelector: '#global-search',
        algoliaOptions: { 'facetFilters': ["version:1.6.0"] },
        debug: true// Set debug to true if you want to inspect the dropdown
    });
    
    $("#search-icon").click(function () {
        $(".trigger").fadeOut("fast", function() {
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

    let timer;
    const toggleDropdown = function (showContent) {
        if (timer) clearTimeout(timer);
        if (showContent) {
            timer = setTimeout(function () {
                $(".gs-version-dropdown").show()
            }, 250);
        } else {
            $(".gs-version-dropdown").hide()
        }
    }

    $("#global-search-dropdown-container")
        .mouseenter(toggleDropdown.bind(null, true))
        .mouseleave(toggleDropdown.bind(null, false))
        .click(function () { $(".gs-version-dropdown").toggle() });

    $("ul.gs-version-dropdown li").each(function() {
        $(this).on("click", function() {
            $("li.gs-opt.active").removeClass("active");
            $(this).addClass("active");
            $("#gs-current-version-label").html(this.innerHTML);
            globalSearch.autocomplete.unbind();
            $(".ds-dropdown-menu").remove();
            globalSearch = docsearch({
                apiKey: '500f8e78748bd043cc6e4ac130e8c0e7',
                indexName: 'apache_mxnet',
                inputSelector: '#global-search',
                algoliaOptions: { 'facetFilters': ["version:" + this.innerHTML] },
                debug: true// Set debug to true if you want to inspect the dropdown
            });
        })
    });
});