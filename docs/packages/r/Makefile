# This is the makefile for compiling Rmarkdown files into the md file with results.
PKGROOT=../../R-package

# ADD The Markdown to be built here, with suffix md
classifyRealImageWithPretrainedModel.md:
mnistCompetition.md:
ndarrayAndSymbolTutorial.md:
fiveMinutesNeuralNetwork.md:

# General Rules for build rmarkdowns, need knitr
%.md: $(PKGROOT)/vignettes/%.Rmd
	rm -rf "../../web-data/mxnet/knitr/$(basename $@)-"*;
	Rscript -e \
	"require(knitr);"\
	"knitr::opts_knit\$$set(root.dir=\".\");"\
	"knitr::opts_chunk\$$set(fig.path=\"../../web-data/mxnet/knitr/$(basename $@)-\");"\
	"knitr::knit(\"$+\")"
