#/bin/bash
# has to be run from root of dato_deps
set -x
set -e 

python_scripts=deps/conda/bin
if [[ $OSTYPE == msys ]]; then
  python_scripts=deps/conda/bin/Scripts
fi

function make_windows_exec_link {
  targetname=$1/`basename $2 .exe`
  echo "#!/bin/sh" > $targetname
  echo "$2 \$@" >> $targetname
}

# conda on windows puts everything in the bin/Scripts folder. This 
# remaps everything in the bin/Scripts folder over to the bin folder where
# we can find them
function windows_create_bin_links {
if [[ $OSTYPE == msys ]]; then
        for i in `ls $PWD/deps/conda/bin/Scripts/*.exe`; do
                make_windows_exec_link "$PWD/deps/conda/bin" "$i"
        done
fi
}

function windows_patch_python_header {
if [[ $OSTYPE == msys ]]; then
        echo "#include <math.h>" > tmp
        cat deps/conda/include/python2.7/pyconfig.h >> tmp
        cp tmp deps/conda/include/python2.7/pyconfig.h
        rm tmp
fi
}


function download_file {
  # detect wget
  echo "Downloading $2 from $1 ..."
  if [ -z `which wget` ] ; then
    if [ -z `which curl` ] ; then
      echo "Unable to find either curl or wget! Cannot proceed with
            automatic install."
      exit 1
    fi
    curl $1 -o $2
  else
    wget $1 -O $2
  fi
} # end of download file

haspython=0
if [ -e deps/conda/bin/python ]; then
        haspython=1
fi

if [ -e deps/conda/bin/python.exe ]; then
        haspython=1
fi
if [[ $haspython == 0 ]]; then
        if [[ $OSTYPE == darwin* ]]; then
                if [ ! -e miniconda.sh ]; then
                        download_file http://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh miniconda.sh
                fi
                bash ./miniconda.sh -p $PWD/deps/conda -b
        elif [[ "$OSTYPE" == "msys" ]]; then
                if [ ! -e miniconda.exe ]; then
                        download_file http://repo.continuum.io/miniconda/Miniconda-latest-Windows-x86_64.exe miniconda.exe
                fi
                $COMSPEC /C "miniconda.exe /S /RegisterPython=0 /AddToPath=0 /D=`cygpath -w $PWD/deps/conda/bin`"
                mkdir -p $PWD/deps/conda/lib
                cp $PWD/deps/conda/bin/*.dll $PWD/deps/conda/lib
        else
                if [ ! -e miniconda.sh ]; then
                        download_file http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh miniconda.sh
                fi
                bash ./miniconda.sh -p $PWD/deps/conda -b
        fi
fi
$python_scripts/conda install -y --file scripts/conda_requirements.txt
$python_scripts/pip install -r scripts/pip_requirements.txt
# for windows
if [ -e deps/conda/bin/include ]; then
        mkdir -p deps/conda/include/python2.7
        cp deps/conda/bin/include/* deps/conda/include/python2.7
fi

windows_patch_python_header
mkdir -p deps/local/lib
mkdir -p deps/local/include
if [ $OSTYPE == "msys" ]; then
  cp deps/conda/lib/python*.dll deps/local/lib
else
  cp deps/conda/lib/libpython* deps/local/lib
fi
cp -R deps/conda/include/python2.7 deps/local/include
