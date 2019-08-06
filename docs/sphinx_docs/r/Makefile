all: html

SHELL := /bin/bash

RST = $(shell find ./source -name "*.rst")

IPYNB_MARKDOWN = $(shell find ./source/tutorial/ -name "*.ipynb")

OBJ = $($(RST) $(IPYNB_MARKDOWN))

html: # build everything
	cd ./source/ ;\
	Rscript ./scripts/Rdoc2SphinxDoc.R ;\
	cd ./tutorial/ ;\
	bash scripts/convertNotebooks.sh ;\
	mkdir mxnetRtutorials ;\
	mkdir ./mxnetRtutorials/data ;\
	mkdir ./mxnetRtutorials/Inception ;\
	cp *.ipynb ./mxnetRtutorials/ ;\
	cp ./data/* ./mxnetRtutorials/data/ ;\
	cp ./Inception/* ./mxnetRtutorials/Inception/ ;\
	zip -r mxnetRtutorials.zip mxnetRtutorials ;\
	rm -rf mxnetRtutorials ;\
	cd ../.. ;\
	sphinx-build -b html -j 8 source/ build/ ;\
	cp ./source/tutorial/mxnetRtutorials.zip ./build/tutorial/mxnetRtutorials.zip


htmlonly: # depends on .rst files and .ipynb files.
	sphinx-build -b html  -j 8  source/ build/

documentation: # depends on man/ files.
	cd source/ ;\
	Rscript ./scripts/Rdoc2
	Doc.R

tutorials: # depends on .md files
	cd source/tutorial/ ;\
	bash scripts/convertNotebooks.sh

clean:
	rm -rf ./build/* ;\
	rm -f ./source/api/* ;\
	rm -f ./source/tutorial/*.ipynb ;\
	rm -f ./source/tutorial/*.params ;\
	rm -f ./source/tutorial/*.json ;\
	rm -f ./source/tutorial/*.txt ;\
	rm -f ./source/tutorial/*.csv ;\
	rm -f ./source/tutorial/*.ndarrays ;\
	rm -f ./source/tutorial/mxnetRtutorials.zip ;\
        rm -rf ./source/tutorial/data/* ;\
	rm -rf ./source/tutorial/Inception ;\
	rm -f ./source/tutorial/Inception.zip
