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

/* Copy code to clipboard */


function addBtn() {
    copyBtn = '<button type="button" class="copy-btn" data-placement="bottom" title="Copy to clipboard">copy</button>'
    codeBlock = $('figure,highlight, div.highlight');
    codeBlock.css('position', 'relative')
    codeBlock.prepend(copyBtn);
    codeBlock.hover(
        function () {
            $(this).children().first().show();
        }, function () {
            $(this).children().first().hide();
        }
    );

};

function html2clipboard(content) {
    var tmpEl = document.createElement("div");
    tmpEl.style.opacity = 0;
    tmpEl.style.position = "absolute";
    tmpEl.style.pointerEvents = "none";
    tmpEl.style.zIndex = -1;

    tmpEl.innerHTML = content;
    document.body.appendChild(tmpEl);

    var range = document.createRange();
    range.selectNode(tmpEl);
    window.getSelection().addRange(range);
    document.execCommand("copy");
    document.body.removeChild(tmpEl);
}

$(document).ready(function () {
    addBtn()

    clipboard = new Clipboard('.copy-btn', {
        target: function (trigger) {
            return trigger.parentNode.querySelector('code');
        }
    });

    clipboard.on('success', function (e) {
        //Deal with codes with leading gap
        var btnClass = e.trigger.classList;
        var lang = btnClass[btnClass.length - 1];
        var lines = e.text.split('\n');
        var hasGap = false;
        var continueSign = '...';

        e.clearSelection();

        for (var i = 0; i < lines.length; ++i) {
            lines[i] = lines[i].replace(/^\s+|\s+$/g, "");
            if (!hasGap && lines[i].startsWith(LANG_GP[lang])) hasGap = true;
        }

        if (hasGap) {
            var content = '';
            for (var i = 0; i < lines.length; ++i) {
                if (lines[i].startsWith(LANG_GP[lang]) || ((lang == 'python' || lang == 'default') &&
                    lines[i].startsWith(continueSign))) {
                    content = content.concat(lines[i].substring(LANG_GP[lang].length, lines[i].length) + '<br />');
                } else if (lines[i].length == 0) content = content.concat('<br />');
            }
            content = content.substring(0, content.length - 6);
            html2clipboard(content);
        }
    });

    clipboard.on('error', function (e) {
        $(e.trigger).attr('title', 'Copy failed. Try again.')
            .tooltip('fixTitle')
            .tooltip('show');
    });
});
