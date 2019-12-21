FROM puckel/docker-airflow:1.10.6

ENV EXECUTOR=Local

USER root

WORKDIR /usr/local/airflow/

COPY requirements.txt .

# RUN pip install --user -r requirements.txt
RUN pip install --requirement requirements.txt

RUN apt-get update && \
    apt-get install -y vim git

USER airflow

EXPOSE 8080
