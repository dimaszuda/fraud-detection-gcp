from kfp import compiler
import google.cloud.aiplatform as aiplatform
from deployment.pipeline import create_pipeline
from models.schema import (
    bq_schema, 
    algoritma_model, 
    metrics_evaluation,
    metric_names
)

from secret import (
    JOB_ID,
    PROJECT_ID,
    REGION_NAME,
    GCS_BUCKET_URI,
    SERVICE_ACCOUNT,
    PIPELINE_NAME,
    TEMPLATE_PATH,
    DATASET_ID,
    TABLE_ID,
)

bq_schema_json = [
    {"name": field.name, "field_type": field.field_type, "mode": field.mode}
    for field in bq_schema
]

PIPELINE_PARAMS = {
        "PROJECT_ID": PROJECT_ID,
        "dataset_id": DATASET_ID,
        "table_id": TABLE_ID,
        "gcs_uri": GCS_BUCKET_URI,
        "bq_schema": bq_schema_json,
        "algoritma": algoritma_model,
        "metrics": metrics_evaluation,
        "metrics_names": metric_names
}

# Compile the pipeline first
compiler.Compiler().compile(
    pipeline_func=create_pipeline,
    package_path=TEMPLATE_PATH
)
print("Bismillah jadi")

try:
    # Now initialize Vertex AI and submit the pipeline job
    aiplatform.init(project=PROJECT_ID, location=REGION_NAME)

    pipeline_ = aiplatform.pipeline_jobs.PipelineJob(
        enable_caching=False,
        display_name=PIPELINE_NAME,
        template_path=TEMPLATE_PATH,
        job_id=JOB_ID,
        parameter_values=PIPELINE_PARAMS)
    print("Successfullly create a pipeline job")

    print("Submit Pipeline Job")
    pipeline_.submit(service_account=SERVICE_ACCOUNT)
    print("Successfully submit pipeline job")
except Exception as e:
    print(f"Error submit pipeline job: {e}")
    raise e