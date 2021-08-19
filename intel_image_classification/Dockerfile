FROM ubuntu:20.04

MAINTAINER "Ritik Jain <ritikjain51@gmail.com>"

WORKDIR /app

# Copy Files for the installation process
COPY jit_model.pt /app/jit_model.pt
COPY serve.py /app/serve.py
COPY model.py /app/model.py
COPY index_to_name.json /app/index_to_name.json
COPY requirements.txt /app/requirements.txt


# EXPOSE PORTS
EXPOSE 8080 8081 8082 7071 7070

# Install python and other requirements
RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-get install python3.8 python3-pip -y
RUN pip3 install -r requirements.txt

# Install JAVA SDK
RUN apt-get install -y openjdk-11-jdk

# Installing torch serve
RUN apt-get install git -y
RUN git clone https://github.com/pytorch/serve.git --single-branch -b master
RUN pip3 install serve/

RUN torch-model-archiver --model-name resnet34_image_classifier --version 1.0 --model-file model.py --serialized-file jit_model.pt --handler serve.py --requirements-file requirements.txt --extra-files index_to_name.json -f

CMD torchserve --model-store . --models resnet34_image_classifier.mar --start