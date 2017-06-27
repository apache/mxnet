/*Copy code to clipboard*/
LANG_GP = {'default':'>>> ', 'python':'>>> ' , 'scala':'scala>', 'julia':'julia> ', 'r':'> ', 'perl':'pdl>' , 'cpp':'', 'bash':'$ '};

function addBtn() {
    copyBtn = '<button type="button" class="btn btn-primary copy-btn" data-toggle="tooltip"' +
              'data-placement="bottom" title="Copy to clipboard"><i class="fa fa-copy"></i></button>'
    for (var lang in LANG_GP) {
        codeBlock = $('div .highlight-' + lang);
        codeBlock.prepend(copyBtn);
        codeBlock.find('.copy-btn').addClass(lang);
        codeBlock.hover(
          function() {
            $(this).children().first().show();
          }, function() {
            $(this).children().first().hide();
          }
        );
    }
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

$(document).ready(function(){
    addBtn()
    $('[data-toggle="tooltip"]').tooltip();
    $('.copy-btn').hover(
      function() {}, function() {
        $(this).attr('title', 'Copy to clipboard').tooltip('fixTitle');
      }
    );

    clipboard = new Clipboard('.copy-btn', {
        target: function(trigger) {
            return trigger.parentNode.querySelector('.highlight');
        }
    });

    clipboard.on('success', function(e) {
        //Deal with codes with leading gap
        var btnClass = e.trigger.classList;
        var lang = btnClass[btnClass.length - 1];
        var lines = e.text.split('\n');
        var hasGap = false;
        var continueSign = '...';

        e.clearSelection();

        for(var i = 0; i < lines.length; ++i) {
            lines[i] = lines[i].replace(/^\s+|\s+$/g, "");
            if(!hasGap && lines[i].startsWith(LANG_GP[lang])) hasGap = true;
        }

        if(hasGap) {
            var content = '';
            for(var i = 0; i < lines.length; ++i) {
                if(lines[i].startsWith(LANG_GP[lang]) || ((lang == 'python' || lang == 'default') &&
                                                          lines[i].startsWith(continueSign))) {
                    content = content.concat(lines[i].substring(LANG_GP[lang].length, lines[i].length) + '<br />');
                }
                else if(lines[i].length == 0) content = content.concat('<br />');
            }
            content = content.substring(0, content.length - 6);
            html2clipboard(content);
        }
        $(e.trigger).attr('title', 'Copied')
             .tooltip('fixTitle')
             .tooltip('show');
    });

    clipboard.on('error', function(e) {
        $(e.trigger).attr('title', 'Copy failed. Try again.')
             .tooltip('fixTitle')
             .tooltip('show');
    });
});
