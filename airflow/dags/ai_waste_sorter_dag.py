"""
Airflow DAG orchestrating the DVC-managed ML pipeline for the AI Waste Sorter.

Tasks:
1. Pull latest dataset/model artifacts from the configured DVC remote.
2. Reproduce the DVC pipeline (train -> promote best weights -> evaluate).
3. Push new artifacts/metrics back to the remote.
"""

from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

REPO_ROOT = Path(__file__).resolve().parents[2].as_posix()

default_args = {
    "owner": "ai-waste-sorter",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="ai_waste_sorter_ci_ct_cd",
    description="CI/CT/CD orchestration for the AI Waste Sorter via DVC",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["waste-sorter", "dvc", "mlops"],
    params={"repo_root": REPO_ROOT},
) as dag:
    pull_dataset = BashOperator(
        task_id="pull_dvc_data",
        bash_command="cd {{ params.repo_root }} && dvc pull waste-detection.dvc",
    )

    run_pipeline = BashOperator(
        task_id="run_dvc_pipeline",
        bash_command="cd {{ params.repo_root }} && dvc repro evaluate",
    )

    push_artifacts = BashOperator(
        task_id="push_dvc_artifacts",
        bash_command="cd {{ params.repo_root }} && dvc push",
    )

    notify = BashOperator(
        task_id="notify_success",
        bash_command='echo "AI Waste Sorter pipeline completed successfully."',
    )

    pull_dataset >> run_pipeline >> push_artifacts >> notify

