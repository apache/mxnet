# This is the makefile for compiling Rmarkdown files into the md file with results.
PKGROOT=../../R-package

# ADD The Markdown to be built here
classifyRealImageWithPretrainedModel.md:
mnistCompetition.Rmd:
ndarrayAndSymbolTutorial.Rmd:

# General Rules for build rmarkdowns, need knitr
%.md: $(PKGROOT)/vignettes/%.Rmd
	rm -rf "../doc-image/mxnet/knitr/$(basename $@)-"*;
	Rscript -e \
	"require(knitr);"\
	"knitr::opts_knit\$$set(root.dir=\".\");"\
	"knitr::opts_chunk\$$set(fig.path=\"../web-data/mxnet/knitr/$(basename $@)-\");"\
	"knitr::knit(\"$+\")"
