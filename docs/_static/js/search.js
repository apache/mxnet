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

/* Display functionality for the search feature */
$(document).ready(function () {
    var searchForm = $("#search-input-wrap").children("form").first();
    searchForm.append('<div class="form-group searchBtn"><input type="submit" class="form-control" value="Go"></div>');
    searchForm.children("div").first().addClass("searchBox");
    $(".searchBox").addClass("searchBoxNorm");

    $('#searchIcon').click(function () {
        if($('#search-input-wrap').is(':hidden')) {
            $('#search-input-wrap').show();
            $('#searchIcon span').removeClass('glyphicon-search');
            $('#searchIcon span').addClass('glyphicon-remove-circle');
        }
        else {
            $('#search-input-wrap').hide();
            $('#searchIcon span').removeClass('glyphicon-remove-circle');
            $('#searchIcon span').addClass('glyphicon-search');
        }
    });
});
