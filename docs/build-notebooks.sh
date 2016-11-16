#!/bin/bash

# scripts to add https://github.com/dmlc/mxnet-notebooksa into docs/

notebook2markdown() {
    src=$1
    dst=$2
    for f in $src/*.ipynb; do
        jupyter nbconvert $f --to markdown
        bname=$(basename "$f" .ipynb)
        echo "\n\nThis page is converted from [${bname}.ipynb](https://github.com/dmlc/$f)." >>${src}/${bname}.md
        mv -f ${src}/${bname}.md ${dst}
        if [ -e ${src}/${bname}_files ]; then
            mv -f ${src}/${bname}_files ${dst}/
        fi
    done
}

rm -rf mxnet-notebooks
git clone https://github.com/dmlc/mxnet-notebooks
notebook2markdown mxnet-notebooks/python/basic tutorials/python/
notebook2markdown mxnet-notebooks/python/how_to how_to/
notebook2markdown mxnet-notebooks/python/tutorials tutorials/python/
rm -rf mxnet-notebooks
