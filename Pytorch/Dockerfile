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
  && python3.7 -m pip install torch \
  && python3.7 -m pip install torchvision \
  && python3.7 -m pip install Pillow \
  && python3.7 -m pip install numpy \
  && python3.7 -m pip install requests

COPY api.py api.py
COPY uvalandmarkmodel.pth uvalandmarkmodel.pth
ENTRYPOINT ["python3.7","api.py"]

