import click
import mlflow.pyfunc
import pandas as pd

from flask import Flask, request, jsonify
from json import dumps
from kafka import KafkaProducer

app = Flask(__name__)

# Default value
KAFKA_PRODUCER = None
PREDICT_MODEL = None
KAFKA_TOPIC = 'bracelet-feed'

# Interpret target values
TARGET_LABELS = {0: "< 50% diameter narrowing", 1: "> 50% diameter narrowing"}

# Use @app.route decorator to register run_prediction as endpoint on /predict/bracelet
@app.route("/predict/bracelet", methods=['POST'])
def run_prediction():
    jsonReq = request.json

    allowedColumns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    # Check paramiters
    if allowedColumns != list(jsonReq.keys()):
        return jsonify({"error": "not allowed collumns"})

    # Publish on kafka
    KAFKA_PRODUCER.send(KAFKA_TOPIC, value=jsonReq)

    # Create the DataFrame as model input
    df = pd.DataFrame({k: [v] for k, v in jsonReq.items()})

    print('\nMake prediction on:\n%s' % df)

    # Run prediction
    prediction = PREDICT_MODEL.predict(df)[0]

    # Interpret prediction value
    predictionLabel = TARGET_LABELS[prediction]

    print('Prediction:\n%s - %s' % (prediction, predictionLabel))

    return jsonify({"prediction": str(prediction), "predictionLabel": predictionLabel})


@click.command(help="Serve the model by a REST API interface")
@click.option("--run-id", default='7f537cbc82ec405ba5cf9df73cc79871')
@click.option("--model-uri", default='model')
@click.option("--kafka-topic", default=KAFKA_TOPIC)
def main(run_id, model_uri, kafka_topic):

    global KAFKA_PRODUCER, KAFKA_TOPIC, PREDICT_MODEL

    KAFKA_TOPIC = kafka_topic

    # Init kafka producer
    # encode payload in json format
    KAFKA_PRODUCER = KafkaProducer(bootstrap_servers=['kafka:9092'],
                                     value_serializer=lambda x: dumps(x).encode('utf-8'))

    # Load prediction model
    PREDICT_MODEL = mlflow.pyfunc.load_model("runs:/{}/{}".format(run_id, model_uri))

    # Start web-server
    app.run(host='0.0.0.0', port=5000, debug=True)


# Run app
if __name__ == "__main__":
    main()


