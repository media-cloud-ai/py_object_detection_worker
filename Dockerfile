FROM mediacloudai/py_mcai_worker_sdk:v0.11.7-media

WORKDIR /sources

ADD . /sources

RUN apt-get update && \
    apt-get install -y \
      python3-opencv

ENV PYTHON_WORKER_FILENAME=/sources/object_detection.py
ENV AMQP_QUEUE=job_object_detection

CMD py_mcai_worker_sdk
