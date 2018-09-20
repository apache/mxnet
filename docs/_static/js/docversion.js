$(document).ready(function () {
	function setVersion(){
  		console.log('Applying current file to links: ', doc[1]);
  		if (document.getElementById('dropdown-menu-position-anchor-version')) {
          		console.log('found the versions element');
          		versionNav = $('#dropdown-menu-position-anchor-version a.main-nav-link');
          		console.log(versionNav);
          		$(versionNav).each( function( index, el ) {
                  		currLink = $( el ).attr('href');
                  		console.log('current link: ', currLink);
                  		version = currLink.match(/\/versions\/([0-9.]+)\//);
                  		if (version) {
                          		console.log('version: ', version[1]);
                          		versionedDoc = '/versions/' + version[1] + '/' + doc[1];
                          		console.log('versionedDoc: ', versionedDoc);
                          		$( el ).attr('href', versionedDoc);
                  		}
          		});
  		}
	}
	let doc = window.location.pathname.match(/^\/versions\/[^*]+\/(api\/.*)$/);
	if (doc) {
  		setVersion();
	};
});
