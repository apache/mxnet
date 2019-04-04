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
