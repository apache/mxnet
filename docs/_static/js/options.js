var versionSelect   = defaultVersion = 'v1.2.1';
var platformSelect    = 'Linux';
var languageSelect  = 'Python';
var processorSelect = 'CPU';
var environSelect   = 'Pip';

$(document).ready(function () {
    function label(lbl) {
        return lbl.replace(/[ .]/g, '-').toLowerCase();
    }
   
    function urlSearchParams(searchString) {
        let urlDict = new Map();
        let searchParams = searchString.substring(1).split("&");
        searchParams.forEach(function(element) {
            kvPair = element.split("=");
            urlDict.set(kvPair[0], kvPair[1]);
        });
        return urlDict;
    }

    function setSelects(){
        let urlParams = urlSearchParams(window.location.search);
        if (urlParams.get('version'))
            versionSelect = urlParams.get('version');
        $('li a:contains(' + versionSelect + ')').parent().siblings().removeClass('active');
        $('li a:contains(' + versionSelect + ')').parent().addClass('active');
        $('.current-version').html( versionSelect + ' <span class="caret"></span></button>' );
        if (urlParams.get('platform'))
            platformSelect = urlParams.get('platform');
        $('button:contains(' + platformSelect + ')').siblings().removeClass('active');
        $('button:contains(' + platformSelect + ')').addClass('active');
        if (urlParams.get('language'))
            languageSelect = urlParams.get('language');
        $('button:contains(' + languageSelect + ')').siblings().removeClass('active');
        $('button:contains(' + languageSelect + ')').addClass('active');
        if (urlParams.get('processor'))
            processorSelect = urlParams.get('processor');
        $('button:contains(' + processorSelect + ')').siblings().removeClass('active');
        $('button:contains(' + processorSelect + ')').addClass('active');
        if (urlParams.get('environ'))
            environSelect = urlParams.get('environ');
        $('button:contains(' + environSelect + ')').siblings().removeClass('active');
        $('button:contains(' + environSelect + ')').addClass('active');
        showContent();
        if (window.location.href.indexOf("/install/index.html") >= 0) {
            if (versionSelect.indexOf(defaultVersion) >= 0) {
                history.pushState(null, null, '/install/index.html?platform=' + platformSelect + '&language=' + languageSelect + '&processor=' + processorSelect);
            } else {
                history.pushState(null, null, '/install/index.html?version=' + versionSelect + '&platform=' + platformSelect + '&language=' + languageSelect + '&processor=' + processorSelect);
            }
        } 
    }

    function showContent() {
        $('.opt-group .opt').each(function(){
            $('.'+label($(this).text())).hide();
            $('.highlight-'+label($(this).text())).hide();
        });
        $('.opt-group .active').each(function(){
            $('.'+label($(this).text())).show();
            $('.highlight-'+label($(this).text())).show();
        });
    }
    showContent();
    setSelects();
    function setContent() {
        var el = $(this);
        let urlParams = urlSearchParams(window.location.search);
        el.siblings().removeClass('active');
        el.addClass('active');
        if ($(this).hasClass("versions")) {
            $('.current-version').html( $(this).text() + ' <span class="caret"></span></button>' );
            if ($(this).text().indexOf(defaultVersion) < 0) {
                if (window.location.search.indexOf("version") < 0) {
                    history.pushState(null, null, '/install/index.html' + window.location.search.concat( '&version=' + $(this).text() ));
                } else {
                    history.pushState(null, null, '/install/index.html' + window.location.search.replace( urlParams.get('version'), $(this).text() ));
                }
            } else if (window.location.search.indexOf("version") >= 0) {
                  history.pushState(null, null, '/install/index.html' + window.location.search.replace( 'version', 'prev' ));
              }
        }
        else if ($(this).hasClass("platforms")) {
            history.pushState(null, null, '/install/index.html' + window.location.search.replace( urlParams.get('platform'), $(this).text() ));
        }
        else if ($(this).hasClass("languages")) {
            history.pushState(null, null, '/install/index.html' + window.location.search.replace( urlParams.get('language'), $(this).text() ));
        }
        else if ($(this).hasClass("processors")) {
            history.pushState(null, null, '/install/index.html' + window.location.search.replace( urlParams.get('processor'), $(this).text() ));
        }
        showContent();
        //window.location.search = window.location.search.replace( urlParams.get('version'), $(this).text() );
    }
    $('.opt-group').on('click', '.opt', setContent);
});
