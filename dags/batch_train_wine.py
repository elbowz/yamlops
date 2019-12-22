from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

import mlflow

def exec_train():
    mlflow.run('/tmp/emanuele/sklearn_elasticnet_wine', 'main', parameters={"alpha": 0.66}, use_conda=False, experiment_name='wine')

args = {
    'owner': 'Emanuele'
}

dag = DAG('batch_training_wine', description='Simple batch of wine',
          schedule_interval='0 12 * * *',
          default_args=args,
          start_date=datetime(2019, 12, 20), catchup=False)

dummy_operator = DummyOperator(task_id='prepare_dataset', retries=3, dag=dag)

hello_operator = PythonOperator(task_id='wine_train', python_callable=exec_train, dag=dag)

dummy_operator >> hello_operator