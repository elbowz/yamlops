FROM python:3.7.3

ENV FLASK_ENV "development"
ENV FLASK_DEBUG True

RUN mkdir /app
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install click && \
    pip install 'mlflow==1.4.0' && \
    pip install 'kafka-python==1.4.7' && \
    pip install 'scikit-learn==0.22'

COPY rest-prediction.py /app/

CMD [ "python", "./rest-prediction.py" ]

EXPOSE 5000