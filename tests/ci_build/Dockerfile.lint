# For lint test
FROM ubuntu:16.04

# Sudo is not present on ubuntu16.04
RUN apt-get update && apt-get install -y python-pip sudo
RUN pip install cpplint pylint
