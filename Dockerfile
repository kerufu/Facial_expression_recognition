# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow
WORKDIR /Facial_expression_recognition
COPY . .

RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 
CMD ["python3", "main.py"]