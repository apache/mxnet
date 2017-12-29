/* Generate url tracking for each page */
var protocol = location.protocol.concat("//");
var host = protocol.concat(window.location.host);
var path = window.location.pathname;
var pathArr = path.split('/');
var icon = '<i class="fa fa-angle-right" aria-hidden="true"></i>';
var urlTracker = "<ul><li><a href=" + host + ">MXNet</a>" + icon + "</li>";

// Check whether this is another version
var lastUrl = host;
var versionIDX = -1;
for (var i = 1; i < pathArr.length; ++i) {
    lastUrl += '/' + pathArr[i];
    if(pathArr[i] == 'versions') {
        versionIDX = i;
        lastUrl += '/' + pathArr[i + 1];
        break;
    }
}
if (versionIDX > 0) {
    pathArr = pathArr.slice(versionIDX + 1, pathArr.length);
    urlTracker = "<ul><li><a href=" + lastUrl + "/index.html>MXNet</a>" + icon + "</li>";
}
else lastUrl = host;

for (var i = 1; i < pathArr.length; ++i) {
    pathVal = pathArr[i]
    if (pathVal == 'index.html' || pathVal.length == 0) continue;
    if (pathVal.indexOf('#') != -1) pathVal = pathVal.substring(0, pathVal.indexOf('#'));
    lastUrl += '/' + pathVal;
    if (pathVal.endsWith('.html')) pathVal = pathVal.substring(0, pathVal.length - 5);
    if (i == pathArr.length - 1 || pathArr[i + 1].length == 0 || pathArr[i + 1] == 'index.html') {
        if ( pathVal == 'faq' ){
             pathVal = "FAQ";
        }
        urlTracker += "<li>" + pathVal.replace(/_/g, ' ') + "</li>";
    }
    else {
        // Check whether current folder has index.html.
        // If it doesn't, disable the link.
        $.ajax(lastUrl + '/index.html', {
            type: "GET",
            statusCode: {
                403: function (response) {
                    if (pathVal == 'api') urlTracker += "<li>API" + icon + "</li>";
                    else urlTracker += "<li>" + pathVal.replace(/_/g, ' ') + icon + "</li>";
                },
                404: function (response) {
                    if (pathVal == 'api') urlTracker += "<li>API" + icon + "</li>";
                    else urlTracker += "<li>" + pathVal.replace(/_/g, ' ') + icon + "</li>";
                }
            },
            success: function () {
                item = pathVal == 'ndarray' ? "NDArray" : pathVal;
                urlTracker += "<li><a href=" + lastUrl + '/index.html' + ">" + item.replace(/_/g, ' ') + "</a>" + icon + "</li>";
            }
        });
    }
}
urlTracker += '</ul>';
$('.page-tracker').append(urlTracker);

/* Generate top download btn*/
if ($('div.download-btn').length > 0) {
    var topBtn = $('div.download-btn').clone();
    topBtn.addClass('download-btn-top');
    topBtn.insertAfter(".page-tracker");
}

/* Adjust footer position */
var footerHeight = 252;
if ($('div.content-block').height() > $(window).height() - footerHeight) {
    $('div.footer').css('position', 'relative');
}
