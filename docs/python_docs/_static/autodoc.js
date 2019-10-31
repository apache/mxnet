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

/* Customizations to the Sphinx auto module plugin output */
function auto_index() {
    var targets = $("dl.class>dt,dl.function>dt");
    var li_node = $("li.current>span>a.current.reference.internal").parent().parent();
    var html = "<ul id='autodoc'>";
    if (li_node.length > 0) {
        if (targets.length > 0) {
            for (var i = 0; i < targets.length; ++i) {
                var id = $(targets[i]).attr('id');
                if (id) {
                    var paths = id.split('.')
                    if (paths.length >= 2) {
                        var id_simple = paths.pop();
                        id_simple = paths.pop() + "." + id_simple;
                    } else {
                        var id_simple = id;
                    }
                    html += "<li><span class='link-wrapper'><a class='reference internal' href='#";
                    html += id;
                    html += "'>" + id_simple + "</a></span</li>";
                }
            }
            html += "</ul>";
            li_node.append(html);
            li_node.prepend("<a><span id='autodoc_toggle' onclick='$(\"#autodoc\").toggle()'>[toggle]</span></a>")
        }
    } else {
        setTimeout(auto_index, 500);
    }

}
$(document).ready(auto_index);