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

$(document).ready(function () {
  // Omitted prompts in Regex for each language
  const LANG_GP = {
    default: [">>> ", "\\.\\.\\."],
    python: [">>> ", "\\.\\.\\."],
    scala: ["scala>"],
    java: [],
    julia: ["julia> "],
    r: ["> "],
    perl: ["pdl>"],
    cpp: [""],
    bash: ["\\$ "],
  };

  // Append a copy button to each code block on the page
  $("figure.highlight, div.highlight").each(function () {
    const copyBtn = $('<button type="button" class="copy-btn">copy</button>');
    $(this)
      .css("position", "relative")
      .prepend(copyBtn)
      .hover(
        () => copyBtn.show(),
        () => copyBtn.hide()
      );
  });

  // Clipboard feature based on Clipboard.js v2.0.6
  const cleanPrompt = function (line, prompts) {
    let res = line;
    for (let i = 0; i < prompts.length; i++) {
      let reg = new RegExp("(?<=^\\s*)" + prompts[i]);
      if (reg.test(res)) {
        res = res.replace(reg, "");
        break;
      }
    }
    return res + "\n";
  };

  const clipboard = new ClipboardJS(".copy-btn", {
    text: function (trigger) {
      const lang = trigger.nextElementSibling.children[0].dataset.lang ||
        trigger.parentNode.parentNode.classList[0].split("-")[1] ||
        "default";
      const langPrompts = LANG_GP[lang] || [];
      const lines = trigger.parentNode.querySelector("code").textContent.split("\n");
      const cleanedCode = lines.reduce(
        (content, line) => content.concat(cleanPrompt(line, langPrompts)),
        "");
      return cleanedCode.replace(/\n$/, "");
    },
  });

  clipboard.on("success", (e) => e.clearSelection());
});
