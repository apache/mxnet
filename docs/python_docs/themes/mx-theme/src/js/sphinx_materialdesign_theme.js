import "../scss/sphinx_materialdesign_theme.scss";
import "material-design-lite";
import "babel-polyfill";
import ScrollSpy from "./scrollspy";
import AdjustHeight from "./adjust-height";

$(function() {

    function reconstructionDrawerGlobalToc() {
        const $globaltoc = $('.mdl-layout__drawer nav');
        const $lists = $globaltoc.find('li');
        $.each($lists, function(index, li) {
            const $li = $(li);
            const $linkWrapper = $('<span class="link-wrapper"></span>');
            const $link = $li.children('a');
            $li.append($linkWrapper.append($link));

            const isCurrent = $li.hasClass('current') && !$link.hasClass('current');
            const $ul = $li.children('ul');
            if ($ul.length) {
                const ulId = `globalnav-${index}`;
                $ul.attr('id', ulId);
                $ul.addClass('collapse');
                const $toggleWrapper = $('<span class="nav-toggle"></span>');
                if (isCurrent) {
                    $ul.addClass('show');
                    $toggleWrapper.addClass('show');
                } else {
                    $ul.hide();
                }

                $li.append(
                    $linkWrapper.append(
                        $toggleWrapper.append(
                            $(`<a class="mdl-button mdl-js-button mdl-button--icon" data-toggle="#${ulId}"><span style="color: #888"><i class="material-icons">keyboard_arrow_down</i></span></span>`)
                        )
                    )
                ).append($ul);
            }
        });
    }

    function collapse() {
        $('.mdl-layout__drawer nav .nav-toggle a').click(function() {
            const $toggle = $(this);
            const id = $toggle.attr('data-toggle');
            $(`ul${id}`).toggleClass('show').animate({height: "toggle", opacity: "toggle"});
            $toggle.parent().toggleClass('show');
        });
    }

    function styleMdlCodeBlock() {
        $('pre').hover(function() {
            $(this).attr('click-to-copy', 'click to copy...');
        });
        $('pre').click(function(){
            var result = copyClipboard(this);
            if (result) {
                $(this).attr('click-to-copy', 'copied!');
            }
        });
    }

    function copyClipboard(selector) {
        var body = document.body;
        if(!body) return false;

        var $target = $(selector);
        if ($target.length === 0) { return false; }

        var text = $target.text();
        var textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        var result = document.execCommand('copy');
        document.body.removeChild(textarea);
        return result;
    }

    function quickSearchClickEvent() {
        const $breadcrumb = $('.breadcrumb');

        $('#waterfall-exp').focus(() => {
            if ($(window).width() <= 1024) {
                $breadcrumb.hide();
            }
        }).blur(() => {
            if ($(window).width() <= 1024) {
                $breadcrumb.show();
            }
        });
    }

    // styleMdlCodeBlock();

    reconstructionDrawerGlobalToc();
    collapse();
    quickSearchClickEvent();


    const spy = new ScrollSpy({
        contentSelector: '.page-content .section',
        navSelector: '.localtoc a',
        scrollSelector: 'main' ,
        className: 'current',
        offsetTop: 64});

    $('.mdl-layout__content').focus();

    $('.mx-card').each(function(){
        $(this).addClass('mdl-card mdl-shadow--2dp');
    });
    $('.mx-card .mx-card-title').each(function(){
        $(this).addClass('mdl-card__title');
    });
    $('.mx-card .mx-card-text').each(function(){
        $(this).addClass('mdl-card__supporting-text');
    });
    $('.mx-card-link').each(function(){
        $(this).hide();
    });
    $('.mdl-card').each(function(){
        $(this).click(function() {
            var url = $(this).find('.mx-card-link').text();
            if (url) {
                window.location = url;
            }
            return true;
        });
    });

    $('a.download').each(function() {
        // button
        var button = document.createElement('button');
        button.className = 'download mdl-button mdl-js-button mdl-button--fab mdl-js-ripple-effect';

        // icon
        var icon = document.createElement('i');
        icon.className = 'material-icons';
        var text = document.createTextNode('file_download');
        icon.appendChild(text);
        button.appendChild(icon);

        // link
        var link = $(this).attr('href');
        button.onclick = function() {
            window.location = link;
        };
        var fileName = link.split("/").slice(-1).pop();
        if (fileName) {
            button.id = fileName.replace('.', '-');
        } else {
            button.id = 'download-button-' + $(this).index();
        }

        // hint
        var hint = document.createElement('div');
        hint.className = 'mdl-tooltip';
        hint.setAttribute('data-mdl-for', button.id);
        var hintText = $(this).find('span.pre').map(function() {
            return $(this).text();
        }).get().join(' ');
        hint.innerHTML = hintText;

        componentHandler.upgradeElement(button);
        $(this).remove();
        var header = $('.section h1').first();
        header.append(button);
        header.append(hint);
    });

    $('.mdl-layout').css('visibility', 'visible');

});
