FROM mxnet/python

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



#Copy jupyter notebook startup script
COPY run_jupyter_notebook.sh /

# IPython
EXPOSE 8888

WORKDIR "/sample-notebook"

#Download a sample notebook
RUN wget https://mxnet.incubator.apache.org/tutorials/python/mnist.ipynb
RUN wget https://mxnet.incubator.apache.org/tutorials/python/linear-regression.ipynb
RUN wget https://mxnet.incubator.apache.org/tutorials/python/predict_image.ipynb

#Start the jupyter notebook
ENTRYPOINT ["/bin/bash", "-c", "/run_jupyter_notebook.sh --ip=0.0.0.0 --allow-root"]
