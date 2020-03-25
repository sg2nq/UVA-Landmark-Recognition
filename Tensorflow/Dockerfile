FROM ubuntu:latest
MAINTAINER Sid  
RUN apt-get update \  
  && apt-get install -y python3.7 \
  && apt-get install -y python3-pip \
  && apt-get install -y python3-setuptools \
  && cd /usr/local/bin \  
  && ln -s /usr/bin/python3.7 python \  
  && python3.7 -m pip install --upgrade pip \
  && python3.7 -m pip install flask \
  && python3.7 -m pip install tensorflow \
  && python3.7 -m pip install Pillow \
  && python3.7 -m pip install requests

RUN python3.7 -m pip show tensorflow

COPY api_test.py api_test.py
COPY model.h5 model.h5
ENTRYPOINT ["python3.7","api_test.py"]

