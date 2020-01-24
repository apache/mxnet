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

/* Installation page display functions for install selector.
   This utility allows direct links to specific install instructions.
*/

$(document).ready(function () {
    function label(lbl) {
        lbl = lbl.replace(/[ .]/g, '-').toLowerCase();

        return lbl;
    }

    function urlSearchParams(searchString) {
        let searchDict = new Map();
        let searchParams = searchString.substring(1).split("&");
        searchParams.forEach(function (element) {
            kvPair = element.split("=");
            searchDict.set(kvPair[0], kvPair[1]);
        });
        return searchDict;
    }

    function is_a_match(elem, text) {

        if (label(elem.text()).includes(label(text))) {
            elem.addClass(('active'))
        }
    }

    function setSelects(urlParams) {
        let queryString = '?';
        $('button.opt').removeClass('active');
        if (urlParams.get('version')) {
            versionSelect = urlParams.get('version');
            $('li.versions').removeClass('active');
            $('li.versions').each(function(){is_a_match($(this), versionSelect)});
            $('.current-version').html( versionSelect + ' <span class="caret"></span>' );
            queryString += 'version=' + versionSelect + '&';
        }
        if (urlParams.get('platform')) {
            platformSelect = label(urlParams.get('platform'));
            $('button.opt').each(function(){is_a_match($(this), platformSelect)});
            queryString += 'platform=' + platformSelect + '&';
        }
        if (urlParams.get('language')) {
            languageSelect = label(urlParams.get('language'));
            $('button.opt').each(function(){
                if (label($(this).text()) === label(languageSelect)) {
                    $(this).addClass(('active'))
                }
            });
            queryString += 'language=' + languageSelect + '&';
        }
        if (urlParams.get('processor')) {
            processorSelect = label(urlParams.get('processor'));
            $('button.opt').each(function(){is_a_match($(this), processorSelect)});
            queryString += 'processor=' + processorSelect + '&';
        }
        if (urlParams.get('environ')) {
            environSelect = label(urlParams.get('environ'));
            $('button.opt').each(function(){is_a_match($(this), environSelect)});
            queryString += 'environ=' + environSelect + '&';
        }
        if (urlParams.get('iot')) {
            iotSelect = label(urlParams.get('iot'));
            $('button.opt').each(function(){is_a_match($(this), iotSelect)});
            queryString += 'iot=' + iotSelect + '&';
        }

        showContent();

        if (window.location.href.indexOf("/get_started") >= 0) {
            history.pushState(null, null, queryString);
        }
    }

    function showContent() {
        $('.opt-group .opt').each(function () {
            $('.' + label($(this).text())).hide();
        });
        $('.opt-group .active').each(function () {
            $('.' + label($(this).text())).show();
        });
    }

    setSelects(urlSearchParams(window.location.search));

    function setContent() {
        var el = $(this);
        let urlParams = urlSearchParams(window.location.search);
        el.siblings().removeClass('active');
        el.addClass('active');
        if ($(this).hasClass("versions")) {
            $('.current-version').html($(this).text());
            urlParams.set("version", $(this).text());
        } else if ($(this).hasClass("platforms")) {
            urlParams.set("platform", label($(this).text()));
        } else if ($(this).hasClass("languages")) {
            urlParams.set("language", label($(this).text()));
        } else if ($(this).hasClass("processors")) {
            urlParams.set("processor", label($(this).text()));
        } else if ($(this).hasClass("environs")) {
            urlParams.set("environ", label($(this).text()));
        } else if ($(this).hasClass("iots")) {
            console.log($(this));
            urlParams.set("iot", label($(this).text()));
        }
        setSelects(urlParams);
    }

    $('.opt-group').on('click', '.opt', setContent);
    $('.install-widget').css("visibility", "visible");
    $('.install-content').css("visibility", "visible");

});
