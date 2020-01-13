from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator

from libs.cassandra import cassandra_connect, cassandra_insert_bracelet

args = {
    'owner': 'Emanuele',
    'start_date': days_ago(1)
}

dag = DAG('insert_bracelet_in_cassandra',
            description='Insert bracelet data in Cassandra',
            schedule_interval=None,
            default_args=args,
            catchup=False
)

def insert_bracelet(**kwargs):

    value = kwargs['dag_run'].conf

    print(f'Insert "{value}" to Cassandra')

    # Connect and open a session with Cassandra
    session = cassandra_connect(host='cassandra', keyspace='yamlops')

    cassandra_insert_bracelet(session, value)

insert_data = PythonOperator(
    task_id='insert_cassandra',
    provide_context=True,
    python_callable=insert_bracelet,
    dag=dag
)