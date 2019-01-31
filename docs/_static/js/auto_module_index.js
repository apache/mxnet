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
function auto_index(module) {
  $(document).ready(function () {
    // find all classes or functions
    var div_query = "div[class='section'][id='" + module + "']";
    var class_query = div_query + " dl[class='class'] > dt";
    var func_query = div_query + " dl[class='function'] > dt";
    var targets = $(class_query + ',' + func_query);

    var li_node = $("div.sphinxsidebarwrapper li a[href='#" + module + "']").parent();
    var html = "<ul>";

    for (var i = 0; i < targets.length; ++i) {
	var id = $(targets[i]).attr('id');
	if ( id ) {
	    // remove 'mxnet.' prefix to make menus shorter
	    var id_simple = id.replace(/^mxnet\./, '');
	    html += "<li><a class='reference internal' href='#";
	    html += id;
	    html += "'>" + id_simple + "</a></li>";
	}
    }

    html += "</ul>";
    li_node.append(html);
  });
}
