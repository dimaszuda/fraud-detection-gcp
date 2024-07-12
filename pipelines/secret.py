import os
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
JOB_ID = f"training-pipeline-{TIMESTAMP}"
PROJECT_ID = os.getenv("PROJECT_NAME")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION_NAME = os.getenv("REGION_NAME")
FILENAME = os.getenv("FILENAME")
TABLE = os.getenv("TABLE")
GCS_BUCKET_URI = os.getenv("GCS_BUCKET_URI")
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
SERVICE_ACCOUNT = os.getenv("ACCOUNT")
PIPELINE_NAME = os.getenv("PIPELINE_NAME")
ENABLE_CACHING = os.getenv("ENABLE_CACHING")
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")