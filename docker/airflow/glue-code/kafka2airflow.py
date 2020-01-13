import click
import subprocess
from json import loads, dumps
from kafka import KafkaConsumer

# Allow import form parent directory
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from dags.libs.cassandra import cassandra_connect, cassandra_insert_bracelet
from dags.libs.kafka import kafka_consumer

@click.command(help="Glue code for subscribe kafka topic and call Airflow or insert in cassandra")
@click.option("--mode", default='cassandra')
def main(mode):
    # Connect and open a session with Cassandra
    session = cassandra_connect(host='cassandra', keyspace='yamlops')

    # Executed on each new topic message
    def kafka_on_msg(message):
        if mode == 'cassandra':
            print('\nInsert Body to Cassandra')

            cassandra_insert_bracelet(session, message.value)

        elif mode == 'airflow':
            dag_id_to_trigger = 'insert_bracelet_in_cassandra'
            print(f'\nTriggering Airflow DAG "{dag_id_to_trigger}" with Body as context')

            trigger_airflow(dag_id_to_trigger, message.value)

    # Create a Kafka Consumer
    kafka_consumer(server='kafka:9092', topic='bracelet-feed', group_id='kafka2airflow', on_message=kafka_on_msg)

# Trigger Airflow DAG
def trigger_airflow(dag_id, value):

    bashCommand = ['airflow', 'trigger_dag', dag_id, f'-c {dumps(value)}']
    return subprocess.run(bashCommand, stdout=subprocess.PIPE)

if __name__ == '__main__':
    main()


# Allow use of python api to trigger airflow (but is experimantal)
# from airflow.api.common.experimental.trigger_dag import trigger_dag
# run_id='run_kafka_%s_%s-tes' % (message.topic, message.offset)
#
# trigger_dag(dag_id='example_trigger_target_dag', run_id=run_id, conf={'message': 1})

# import requests
#
# res = requests.post('http://localhost:8080/airflow/api/experimental/dags/example_trigger_target_dag/dag_runs',
#                            headers={'Content-Type': 'application/json', 'Cache-Control': 'no-cache'},
#                             json={'conf': {'message': 'ciao'}})