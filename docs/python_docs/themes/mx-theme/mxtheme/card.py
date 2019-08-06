from sphinx.locale import _
from docutils import nodes
from docutils.parsers.rst import Directive, directives

class card(nodes.General, nodes.Element):
    pass

class CardDirective(Directive):

    # defines the parameter the directive expects
    # directives.unchanged means you get the raw value from RST
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'title': directives.unchanged,
                   'link': directives.unchanged,
                   'is_head': directives.unchanged}
    has_content = True
    add_index = False

    def run(self):
        # gives you access to the options of the directive
        options = self.options

        cid = nodes.make_id("card-{}".format(options['title']))

        classes = ['mx-card']
        if options.get('is_head', 'False').lower() == 'true':
            classes.append('head-card')
        container = nodes.container(ids=[cid], classes=classes)

        container += nodes.inline('', options['title'], classes=['mx-card-title'])
        link = options.get('link')
        if link:
            container += nodes.inline('', link, classes=['mx-card-link'])

        para = nodes.paragraph(classes=['mx-card-text'])
        self.state.nested_parse(self.content, self.content_offset, para)
        container += para

        # we return the result
        return [container]
