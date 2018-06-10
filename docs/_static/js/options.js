var versionSelect   = defaultVersion = 'v1.2.0';
var deviceSelect    = 'Linux';
var languageSelect  = 'Python';
var processorSelect = 'CPU';
var environSelect   = 'Pip';

$(document).ready(function () {
    function label(lbl) {
        return lbl.replace(/[ .]/g, '-').toLowerCase();
    }

    function setSelects(){
        let urlParams = new URLSearchParams(window.location.search);
        if (urlParams.get('version'))
            versionSelect = urlParams.get('version');
        $('li a:contains(' + versionSelect + ')').parent().siblings().removeClass('active');
        $('li a:contains(' + versionSelect + ')').parent().addClass('active');
        $('.current-version').html( versionSelect + ' <span class="caret"></span></button>' );
        if (urlParams.get('device'))
            deviceSelect = urlParams.get('device');
        $('button:contains(' + deviceSelect + ')').siblings().removeClass('active');
        $('button:contains(' + deviceSelect + ')').addClass('active');
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
        if (window.location.href.includes("/install/index.html")) {
            if (versionSelect.includes(defaultVersion)) {
                history.pushState(null, null, '/install/index.html?device=' + deviceSelect + '&language=' + languageSelect + '&processor=' + processorSelect);
            } else {
                history.pushState(null, null, '/install/index.html?version=' + versionSelect + '&device=' + deviceSelect + '&language=' + languageSelect + '&processor=' + processorSelect);
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
        let urlParams = new URLSearchParams(window.location.search);
        el.siblings().removeClass('active');
        el.addClass('active');
        if ($(this).hasClass("versions")) {
            $('.current-version').html( $(this).text() + ' <span class="caret"></span></button>' );
            if (!$(this).text().includes(defaultVersion)) {
                if (!window.location.search.includes("version")) {
                    history.pushState(null, null, '/install/index.html' + window.location.search.concat( '&version=' + $(this).text() ));
                } else {
                    history.pushState(null, null, '/install/index.html' + window.location.search.replace( urlParams.get('version'), $(this).text() ));
                }
            } else if (window.location.search.includes("version")) {
                  history.pushState(null, null, '/install/index.html' + window.location.search.replace( 'version', 'prev' ));
              }
        }
        else if ($(this).hasClass("Devices")) {
            history.pushState(null, null, '/install/index.html' + window.location.search.replace( urlParams.get('device'), $(this).text() ));
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
