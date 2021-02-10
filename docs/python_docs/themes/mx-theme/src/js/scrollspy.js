export default class ScrollSpy {
    constructor(args) {

        this.doc = document;
        this.nav = this.doc.querySelectorAll(args.navSelector);

        if(!this.nav.length === 0) { return }

        this.win = window;
        this.winHeight = this.win.innerHeight;

        this.scrollElement = this.doc.querySelector(args.scrollSelector);
        this.className = args.className;
        this.offsetTop = args.offsetTop || 0;

        this.contents = [];
        this.contents = this.getContents(args.contentSelector);

        this.attachEvent();
    }

    attachEvent() {
        let scrollTimer;
        this.scrollElement.addEventListener('scroll', () => {
            if (scrollTimer) {
                clearTimeout(scrollTimer);
            }

            scrollTimer = setTimeout(() => {
                this.spy();
            }, 1);
        });

        let resizeTimer;
        this.scrollElement.addEventListener('resize', () => {
            if (resizeTimer) {
                clearTimeout(resizeTimer);
            }

            resizeTimer = setTimeout(() => {
                this.spy();
            }, 1);
        });

        this.scrollElement.addEventListener("click", (e) => {
            const target = e.target;
            if (target.tagName !== "A") return;
            window.onclickToc = true;
            for (let i = 0, max = this.nav.length; i < max; i++) {
                const navElement = this.nav[i];
                if (navElement.href === target.href) {
                    navElement.classList.add(this.className);
                    navElement.classList.add('mdl-color-text--primary');
                } else {
                    navElement.classList.remove(this.className);
                    navElement.classList.remove('mdl-color-text--primary');
                }
            }
        });
    }

    getContents(contentSelector) {
        const targets = [];
        for (let i = 0, max = this.nav.length; i < max; i++) {
            const href = this.nav[i].href;
            targets.push(this.doc.getElementById(href.split('#')[1]));
        }
        return targets;
    }

    spy() {
        let elements = this.getViewState();
        this.toggleNavClass(elements);
    }

    getViewState() {
        const elementListInView = [];
        for (let i = 0, max = this.contents.length; i < max; i++) {
            const current = this.contents[i];
            if (current && this.isView(current)) {
                elementListInView.push(current);
            }
        }

        return elementListInView;
    }

    isView(element) {
        const scrollTop = this.scrollElement.scrollTop;
        const subHeaderRect = document.querySelector(".mdl-layout__header-row").getBoundingClientRect();
        const headerHeight = subHeaderRect.top + subHeaderRect.height;
        const scrollBottom = scrollTop + window.innerHeight - headerHeight;
        const rect = element.getBoundingClientRect();
        const elementTop = rect.top + scrollTop;
        const elementBottom = elementTop + element.offsetHeight;

        return elementTop < scrollBottom - 30 && elementBottom > scrollTop + headerHeight + 30;
    }

    toggleNavClass(elements) {
        if (window.onclickToc) {
            window.onclickToc = false;
            return;
          }
        let maxDepth = 0;
        let maxDepthElement = $();

        for (let i = 0, max = elements.length; i < max; i++) {
            const el = elements[i];
            const tempDepth = this.getTagDepth(el);
            if (maxDepth < tempDepth) {
                maxDepth = tempDepth;
                maxDepthElement = el;
            }
        }

        for (let i = 0, max = this.nav.length; i < max; i++) {
            const navElement = this.nav[i];
            if (navElement.href.split('#')[1] === maxDepthElement.id) {
                navElement.classList.add(this.className);
                navElement.classList.add('mdl-color-text--primary');
            } else {
                navElement.classList.remove(this.className);
                navElement.classList.remove('mdl-color-text--primary');
            }
        }
    }

    getTagDepth(element) {
        return parseInt($(element).find('h1,h2,h3,h4,h5,h6').get(0).tagName.split('H')[1]);
    }
}
