$(document).ready(function () {
    function label(lbl) {
        return lbl.replace(/[ .]/g, '-').toLowerCase();
    }
    function showContent() {
        $('.opt-group .opt').each(function(){
            $('.'+label($(this).text())).hide();
            $('.highlight-'+label($(this).text())).hide();
        });
        $('.opt-group .active').each(function(){
            $('.'+label($(this).text())).show();
            $('.highlight-'+label($(this).text())).show();
        });
    }
    showContent();
    function setContent() {
        var el = $(this);
        el.siblings().removeClass('active');
        el.addClass('active');
        showContent();
    }
    $('.opt-group').on('click', '.opt', setContent);
});
