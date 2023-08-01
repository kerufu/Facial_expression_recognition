# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu
WORKDIR /fer
COPY ./requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y