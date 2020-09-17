#FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

FROM python:3.7

WORKDIR /usr/src/app

COPY api/requirements.txt ./


RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .


ENV PYTHONPATH .
ENTRYPOINT ["tail", "-f", "/dev/null"]