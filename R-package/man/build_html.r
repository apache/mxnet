"Helper function to generate HTML from .rd files for docs build. Courtesy of http://yihui.name/en/2012/10/build-static-html-help/"
static_help = function(pkg, links = tools::findHTMLlinks()) {
  pkgRdDB = tools:::fetchRdDB(file.path(find.package(pkg), 'help', pkg))
  force(links); topics = names(pkgRdDB)
  for (p in topics) {
    tools::Rd2HTML(pkgRdDB[[p]], paste(p, 'html', sep = '.'),
                   package = pkg, Links = links, no_links = is.null(links))
  }
}

static_help("mxnet")

