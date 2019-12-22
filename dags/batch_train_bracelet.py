from datetime import datetime
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

import mlflow

def exec_train():
    mlflow.run('https://github.com/elbowz/dummy-bracelet4yamlops.git', 'main', parameters={"degree": 3}, use_conda=False, experiment_name='bracelet')

args = {
    'owner': 'Emanuele'
}

dag = DAG('batch_training_bracelet', description='Simple batch of bracelet',
          schedule_interval='0 8 * * *',
          default_args=args,
          start_date=datetime(2019, 12, 20), catchup=False)

dummy_operator = DummyOperator(task_id='prepare_dataset', retries=3, dag=dag)

hello_operator = PythonOperator(task_id='bracelet_train', python_callable=exec_train, dag=dag)

dummy_operator >> hello_operator