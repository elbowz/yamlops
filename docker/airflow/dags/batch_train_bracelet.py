from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

import mlflow


def exec_train():
    mlflow.run('https://github.com/elbowz/dummy-bracelet4yamlops.git', 'main', parameters={"degree": 3}, use_conda=False, experiment_name='bracelet')


args = {
    'owner': 'Emanuele',
    'start_date': days_ago(1)
}

dag = DAG('batch_training_bracelet', description='Simple batch of bracelet',
          schedule_interval='0 8 * * *',
          default_args=args,
          catchup=False)

dummy_operator = DummyOperator(
    task_id='prepare_dataset',
    retries=3,
    dag=dag
)

bracelet_train = PythonOperator(
    task_id='bracelet_train',
    python_callable=exec_train,
    dag=dag
)

dummy_operator >> bracelet_train
