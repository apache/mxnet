# For spell checker
FROM mxnet/aml:latest

RUN yum install -y enchant
RUN pip install pyenchant grammar-check html2text
RUN pip install sphinx==1.5.1 CommonMark==0.5.4 breathe mock==1.0.1 recommonmark
