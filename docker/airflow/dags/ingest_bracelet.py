from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator

from libs.kafka import kafka_consumer
from libs.cassandra import cassandra_connect, cassandra_insert_bracelet

args = {
    'owner': 'Emanuele',
    'start_date': days_ago(1)
}

dag = DAG(
    'ingest_bracelet',
    description='Take data from Kafka topic and put in Cassandra',
    schedule_interval='*/20 * * * *',
    default_args=args,
    catchup=False
)

def broker_consumer(server, topic, group_id, consumer_timeout_ms=None, **kwargs):
    ti = kwargs['ti']

    # Create a Kafka Consumer
    messages = kafka_consumer(server, topic, group_id, consumer_timeout_ms)

    # Push messages for next task
    ti.xcom_push(key='messages', value=messages)


def insert_bracelet(**kwargs):

    ti = kwargs['ti']

    # Pull messages from the previous task
    messages = ti.xcom_pull(key='messages', task_ids='kafka_consumer')

    session = cassandra_connect(host='cassandra', keyspace='yamlops')

    for message in messages:
        print(f'Insert "{message}" to Cassandra')

        cassandra_insert_bracelet(session, message)

broker_consumer = PythonOperator(
    task_id='kafka_consumer',
    python_callable=broker_consumer,
    provide_context=True,
    op_kwargs={'server': 'kafka:9092',
               'topic': 'bracelet-feed',
               'group_id': 'kafka2airflow',
               'consumer_timeout_ms': 4000
               },
    dag=dag
)

insert_data = PythonOperator(
    task_id='insert_cassandra',
    python_callable=insert_bracelet,
    provide_context=True,
    dag=dag
)

broker_consumer >> insert_data
