FROM mxnet/python:gpu

RUN apt-get update && apt-get install -y python-dev && apt-get install -y python3-dev

RUN pip --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        && \
    python -m ipykernel.kernelspec


# Copy sample notebook
COPY sample-notebook /sample-notebook

#Start jupyter notebook script
COPY run_jupyter_notebook.sh /

# IPython
EXPOSE 8888

WORKDIR "/sample-notebook"

ENTRYPOINT ["/bin/bash", "-c", "/run_jupyter_notebook.sh --ip=127.0.0.1 --allow-root"]
