FROM python:3.6

# Install PostgreSQL client
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib

RUN mkdir /root/app/
WORKDIR /root/app/

# Install python dependencies
COPY rafiki/client/requirements.txt client/requirements.txt
RUN pip install -r client/requirements.txt
COPY rafiki/train_worker/requirements.txt train_worker/requirements.txt
RUN pip install -r train_worker/requirements.txt
COPY rafiki/inference_worker/requirements.txt inference_worker/requirements.txt
RUN pip install -r inference_worker/requirements.txt

COPY rafiki/ rafiki/

# Copy init script
COPY scripts/start_worker.py start_worker.py

ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /root/app/

ENTRYPOINT [ "python", "start_worker.py" ]
