from datetime import datetime, timedelta

from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

from src.lab import (
    load_data,
    profile_training_data,
    data_preprocessing,
    build_save_model,
    load_model_elbow,
)

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
# The old airflow.configuration API is deprecated.

MODEL_FILENAME = "lab1_kmeans_model.sav"
LAB_OWNER = "esakkivel"


def print_lab_summary(prediction_report):
    print("Lab 1 summary")
    print(f"Owner: {LAB_OWNER}")
    print(f"Model file: {MODEL_FILENAME}")
    print(f"Test rows: {prediction_report['test_rows']}")
    print(f"Predicted clusters: {prediction_report['predictions']}")
    return {
        "owner": LAB_OWNER,
        "model_file": MODEL_FILENAME,
        "prediction_report": prediction_report,
    }


# Define default arguments for your DAG
default_args = {
    'owner': LAB_OWNER,
    'start_date': datetime(2025, 1, 15),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Lab 1 KMeans clustering workflow',
    schedule=None,
    catchup=False,
    tags=["airflow-lab", "kmeans"],
) as dag:

    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    profile_training_data_task = PythonOperator(
        task_id='profile_training_data_task',
        python_callable=profile_training_data,
        op_args=[load_data_task.output],
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, MODEL_FILENAME],
    )

    # Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=[MODEL_FILENAME, build_save_model_task.output],
    )

    summary_task = PythonOperator(
        task_id='print_lab_summary_task',
        python_callable=print_lab_summary,
        op_args=[load_model_task.output],
    )

    # Set task dependencies
    load_data_task >> profile_training_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task >> summary_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()
