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

/* Installation page display functions for install selector */

$(document).ready(function () {
    function label(lbl) {
        lbl = lbl.replace(/[ .]/g, '-').toLowerCase();

        return lbl;
    }

    function urlSearchParams(searchString) {
        let urlDict = new Map();
        let searchParams = searchString.substring(1).split("&");
        searchParams.forEach(function (element) {
            kvPair = element.split("=");
            urlDict.set(kvPair[0], kvPair[1]);
        });
        return urlDict;
    }

    function is_a_match(elem, text) {

        if (label(elem.text()).includes(label(text))) {
            elem.addClass(('active'))
        }
    }

    function setSelects() {
        let urlParams = urlSearchParams(window.location.search);
        if (urlParams.get('version'))
            versionSelect = urlParams.get('version');
        $('.current-version').html( versionSelect + ' <span class="caret"></span></button>' );
        if (urlParams.get('platform'))
            platformSelect = label(urlParams.get('platform'));
        if (urlParams.get('language'))
            languageSelect = label(urlParams.get('language'));
        if (urlParams.get('processor'))
            processorSelect = label(urlParams.get('processor'));
        if (urlParams.get('environ'))
            environSelect = label(urlParams.get('environ'));

        $('li.versions').removeClass('active');
        $('li.versions').each(function(){is_a_match($(this), versionSelect)});
        $('button.opt').removeClass('active');
        $('button.opt').each(function(){is_a_match($(this), platformSelect)});
        $('button.opt').each(function(){is_a_match($(this), languageSelect)});
        $('button.opt').each(function(){is_a_match($(this), processorSelect)});
        $('button.opt').each(function(){is_a_match($(this), environSelect)});

        showContent();
        if (window.location.href.indexOf("/get_started/") >= 0) {
            history.pushState(null, null, '?version=' + versionSelect + '&platform=' + platformSelect + '&language=' + languageSelect + '&environ=' + environSelect + '&processor=' + processorSelect);
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

    showContent();
    setSelects();

    function setContent() {
        var el = $(this);
        let urlParams = urlSearchParams(window.location.search);
        el.siblings().removeClass('active');
        el.addClass('active');
        if ($(this).hasClass("versions")) {
            $('.current-version').html($(this).text());
            if (window.location.search.indexOf("version") < 0) {
                if (window.location.search.length > 0) {
                    var url = 'index.html' + window.location.search.concat('&version=' + $(this).text());
                } else {
                    var url = 'index.html?version=' + $(this).text();
                }
                history.pushState(null, null, url);
            } else {
                history.pushState(null, null, 'index.html' + window.location.search.replace(urlParams.get('version'), $(this).text()));
            }
        } else if ($(this).hasClass("platforms")) {
            history.pushState(null, null, 'index.html' + window.location.search.replace('='+urlParams.get('platform'), '='+label($(this).text())));
        } else if ($(this).hasClass("languages")) {
            history.pushState(null, null, 'index.html' + window.location.search.replace('='+urlParams.get('language'), '='+label($(this).text())));
        } else if ($(this).hasClass("processors")) {
            history.pushState(null, null, 'index.html' + window.location.search.replace('='+urlParams.get('processor'), '='+label($(this).text())));
        } else if ($(this).hasClass("environs")) {
            history.pushState(null, null, 'index.html' + window.location.search.replace('='+urlParams.get('environ'), '='+label($(this).text())));
        }

        showContent();
    }

    $('.opt-group').on('click', '.opt', setContent);
    $('.install-widget').css("visibility", "visible");
    $('.install-content').css("visibility", "visible");

});
