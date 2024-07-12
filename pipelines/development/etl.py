import os
from typing import List
from google.cloud import bigquery
from dependecies import logging
from models.schema import bq_schema
from kfp.dsl import component, Output, Dataset

BASE_IMAGE = "gcr.io/deeplearning-platform-release/xgboost-cpu"

@component(base_image=BASE_IMAGE)
def extract_table_to_bq(
        PROJECT_ID: str, 
        dataset_id: str, 
        table_id: str, 
        gcs_uri: str,
        dataset: Output[Dataset],
        bq_schema: List) -> None:
    """
    Extract a table from GCS to BigQuery.
    """
    logging.info("Create job to ingest data from gcs to bq")
    try:
        TABLE_ID = f"{PROJECT_ID}.{dataset_id}.{table_id}"
        client = bigquery.Client()
        job_config = bigquery.LoadJobConfig(
            schema=bq_schema,
            source_format=bigquery.SourceFormat.CSV,
            field_delimiter=",",
            skip_leading_rows=1,
        )
        load_job = client.load_table_from_uri(
            gcs_uri,
            TABLE_ID,
            job_config=job_config
        )
        load_job.result()
        destination_table = client.get_table(TABLE_ID)
        print(f"Loaded {destination_table.num_rows} rows.")
        logging.info("Successfully ingest data")
    except Exception as e:
        logging.error(f"Error ingest data: {e}")
        return None
