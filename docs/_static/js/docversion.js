/*!
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/* Set the version of the website */
function setVersion(anchor){
        if (arguments.length==0) {
            anchor = window.location.hash
        };
        let doc = window.location.pathname.match(/^\/versions\/[^\/]+\/([^*]+.*)$/);
        if (doc) {
            if (document.getElementById('dropdown-menu-position-anchor-version')) {
                    versionNav = $('#dropdown-menu-position-anchor-version a.main-nav-link');
                    $(versionNav).each( function( index, el ) {
                            currLink = $( el ).attr('href');
                            version = currLink.match(/\/versions\/([^\/]+)\//);
                            if (version) {
                                    versionedDoc = '/versions/' + version[1] + '/' + doc[1] + (anchor || '') + (window.location.search || '');
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
    setVersion($(this).attr("href"));
});
