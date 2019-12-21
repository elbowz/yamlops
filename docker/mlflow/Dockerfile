FROM python:3.7.0

#RUN mkdir -p /mlflow/artifacts

# psycopg2-binary allow using postgres as backend-store
RUN pip install --upgrade pip && \
    pip install 'psycopg2-binary==2.8.4' && \
    pip install 'mlflow==1.4.0'

EXPOSE 8080
