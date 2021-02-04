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
  const default_version = $("#gs-current-version-label").text() || "master";
  // bind docsearch
  const globalSearch = docsearch({
    apiKey: "500f8e78748bd043cc6e4ac130e8c0e7",
    indexName: "apache_mxnet",
    inputSelector: "#global-search",
    algoliaOptions: {
      facetFilters: ["version:" + default_version],
    },
    debug: false, // Set debug to true if you want to inspect the dropdown
  });

  const globalSearchMobile = docsearch({
    apiKey: "500f8e78748bd043cc6e4ac130e8c0e7",
    indexName: "apache_mxnet",
    inputSelector: "#global-search-mobile",
    algoliaOptions: {
      facetFilters: ["version:" + default_version],
      hitsPerPage: 5,
    },
    debug: false, // Set debug to true if you want to inspect the dropdown
  });

  // search bar animation and event listeners for desktop 
  $("#gs-search-icon").click(function () {
    $(".trigger").fadeOut("fast", function () {
      $("#global-search-form").css("display", "inline-block");
      $("#global-search-close").show();
      $("#global-search-dropdown-container").show();
      $("#global-search")
        .animate({
          width: "300px",
        })
        .focus();
    });
  });

  $("#global-search-close").click(function () {
    $("#global-search-dropdown-container").hide();
    $("#global-search").animate(
      {
        width: "0px",
      },
      function () {
        $(this).hide();
        $("#global-search-form").hide();
        $(".trigger").fadeIn("fast");
      }
    );
  });

  $("#global-search-dropdown-container").click(function (e) {
    $(".gs-version-dropdown").toggle();
    e.stopPropagation();
  });

  $("ul.gs-version-dropdown li").each(function () {
    $(this).on("click", function () {
      $("#global-search").val("");
      $(".gs-version-dropdown li.gs-opt.active").removeClass("active");
      $(this).addClass("active");
      $("#gs-current-version-label").html(this.innerHTML);
      globalSearch.algoliaOptions = {
        facetFilters: ["version:" + this.innerHTML],
      };
    });
  });

  // search bar event listeners for mobile and tablet 
  $("#global-search-dropdown-container-mobile").click(function (e) {
    $(".gs-version-dropdown-mobile").toggle();
    e.stopPropagation();
  });

  $("ul.gs-version-dropdown-mobile li").each(function () {
    $(this).on("click", function () {
      $("#global-search-mobile")
        .val("")
        .attr("placeholder", "v - " + this.innerHTML);
      $(".gs-version-dropdown-mobile li.gs-opt.active").removeClass("active");
      $(this).addClass("active");
      globalSearchMobile.algoliaOptions = {
        facetFilters: ["version:" + this.innerHTML],
        hitsPerPage: 5,
      };
    });
  });

  // Common logic
  $(document).click(function () {
    $(".gs-version-dropdown").hide();
    $(".gs-version-dropdown-mobile").hide();
  });
});
