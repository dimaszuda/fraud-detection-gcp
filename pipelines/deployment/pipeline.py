from development.etl import extract_table_to_bq
from development.create_dataset import create_df
from training.train import train_model
from training.test import test_model
from training.evaluation import calculate_metrics, evaluation_metrics
from typing import List
import json
from secret import PIPELINE_NAME, PIPELINE_ROOT
from kfp import dsl
from models.schema import metric_names, true_column, pred_column

@dsl.pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def create_pipeline(
    PROJECT_ID: str,
    dataset_id: str,
    table_id: str,
    gcs_uri: str,
    bq_schema: List,
    algoritma: List[str],
    metrics: List[str],
    metrics_names: List[str]
) -> None:
    
    etl = extract_table_to_bq(
        PROJECT_ID=PROJECT_ID,
        dataset_id=dataset_id,
        table_id=table_id,
        gcs_uri=gcs_uri,
        bq_schema=bq_schema
    ).set_display_name("Extract table to bq")

    create_data = create_df(
        data_input=etl.outputs['dataset']
    ).set_display_name("Create dataset train and test")

    training_model = train_model(
        training_data=create_data.outputs['dataset_train']
    ).set_display_name("Training Model")

    testing_data = test_model(
        model_names=algoritma,
        test_data=create_data.outputs['dataset_test'],
        model=training_model.outputs['model']
    ).set_display_name("Testing model by create predictions")

    calculating_metrics = calculate_metrics(
        metrics=metrics,
        algoritma=algoritma,
        training_data=create_data.outputs['dataset_train'],
        test_data=create_data.outputs['dataset_test']
    ).set_display_name("Calculating metrics of models")


    eval_metrics = evaluation_metrics(
        predictions=testing_data.outputs['predictions'],
        dict_keys=create_data.outputs['dict_keys'],
        metrics_name=metrics_names,
        model_names=algoritma,
        true_column_name=true_column,
        pred_column_name=pred_column
    ).set_display_name("Evaluation metrics")
