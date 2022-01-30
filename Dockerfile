# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /image-retrieval

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip3 install lshashpy3==0.0.8

COPY . .

WORKDIR /image-retrieval/app

CMD [ "python3", "app.py" , "--large", "faiss" ]