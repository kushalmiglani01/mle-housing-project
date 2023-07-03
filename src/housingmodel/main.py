import os

import mlflow
from housingmodel.default_args import PROJECT_ROOT
from housingmodel.ingest_data import main as download_data
from housingmodel.score import main as evaluation
from housingmodel.train import main as train_model

artifact_path = os.path.join(PROJECT_ROOT, "artifacts")
print(artifact_path)

exp_name = f"end-to-end_experiment"
print(f"file:/{artifact_path}/mlruns")
mlflow.set_tracking_uri(f"file:{artifact_path}/mlruns")
mlflow.set_experiment(exp_name)
with mlflow.start_run(
    experiment_id=mlflow.get_experiment_by_name(exp_name).experiment_id
) as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(
        run_name="CHILD_RUN_DOWNLOAD_DATA",
        experiment_id=mlflow.get_experiment_by_name(exp_name).experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        download_data()
        mlflow.log_param("child", "yes")

    with mlflow.start_run(
        run_name="CHILD_RUN_TRAIN_MODEL",
        experiment_id=mlflow.get_experiment_by_name(exp_name).experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        train_model()
        mlflow.log_param("child", "yes")

    with mlflow.start_run(
        run_name="CHILD_RUN_SCORE_MODEL",
        experiment_id=mlflow.get_experiment_by_name(exp_name).experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        evaluation()
        mlflow.log_param("child", "yes")
