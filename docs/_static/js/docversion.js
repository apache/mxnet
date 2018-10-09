function setVersion(){
        let doc = window.location.pathname.match(/^\/(api\/.*)$/) || window.location.pathname.match(/^\/versions\/[^*]+\/(api\/.*)$/);
        if (doc) {
            if (document.getElementById('dropdown-menu-position-anchor-version')) {
                    versionNav = $('#dropdown-menu-position-anchor-version a.main-nav-link');
                    $(versionNav).each( function( index, el ) {
                            currLink = $( el ).attr('href');
                            version = currLink.match(/\/versions\/([0-9.master]+)\//);
                            if (version) {
                                    versionedDoc = '/versions/' + version[1] + '/' + doc[1] + (window.location.hash || '');
                                    $( el ).attr('href', versionedDoc);
                            }
                    });
            }        
        }
}

$(document).ready(function () {
    setVersion();
});

$('a.reference.internal').click(function(){
    setVersion();
});
