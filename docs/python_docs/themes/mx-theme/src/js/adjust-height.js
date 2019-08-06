export default class AdjustHeight {
    
    constructor() {
        this.header = $('header.mdl-layout__header');
        this.pagenation = $('div.pagenation');
        this.footer = $('footer.mdl-mini-footer');
        this.win = $(window);
        this.scrollElement = $('main');
        this.content = $('.page-content');
        this.outline = $('.side-doc-outline--content');

        this.attachEvent();
        this.adjust();
    }

    adjust() {
        this.setPageContentMinHeight();
        this.setLocaltocHeight();
    }

    setPageContentMinHeight() {
        const winH = this.win.innerHeight();
        const headerHeight = this.header.outerHeight();
        const footerHeight = this.footer.outerHeight(true) + this.pagenation.outerHeight(true);

        this.content.css('min-height', this.win.innerHeight() - headerHeight - footerHeight);
    }

    setLocaltocHeight() {
        const pagenationPotisionTop = this.pagenation.position().top + parseInt(this.pagenation.css('margin-top'), 10);
        const outlineBottom = this.scrollElement.scrollTop() + this.content.outerHeight();
        const winH = this.win.innerHeight() - this.header.outerHeight();

        let min = 0;
        min = outlineBottom > pagenationPotisionTop ? pagenationPotisionTop : outlineBottom;
        min = min > winH ? winH : min;
        this.outline.css('height', min);
    }

    attachEvent() {
        let scrollTimer;
        this.scrollElement.on('scroll', () => {
            if (scrollTimer) {
                clearTimeout(scrollTimer);
            }

            scrollTimer = setTimeout(() => {
                this.adjust();
            }, 1);
        });

        let resizeTimer;
        this.win.on('resize', () => {
            if (resizeTimer) {
                clearTimeout(resizeTimer);
            }

            resizeTimer = setTimeout(() => {
                this.adjust();
            }, 1);
        });
    }
}