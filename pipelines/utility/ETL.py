import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME")
BUCKET_NAME = os.getenv("BUCKET_NAME")
REGION_NAME = os.getenv("REGION_NAME")
FILENAME = os.getenv("FILENAME")
TABLE = os.getenv("TABLE")

class ParseCSV(beam.DoFn):
    def process(self, element):
        fields = element.split(",")
        try:
            yield {
                'type': fields[1],
                'amount': float(fields[2]),
                'oldbalanceOrg': float(fields[4]),
                'newbalanceOrig': float(fields[5]),
                'oldbalanceDest': float(fields[7]),
                'newbalanceDest': float(fields[8]),
                'isFraud': int(fields[9])
            }
        except ValueError as e:
            print(f"Error converting record: {element}, error: {e}")
    
def main(argv=None):
    pipeline_options = PipelineOptions()
    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    google_cloud_options.project = PROJECT_NAME
    google_cloud_options.job_name = 'dataflow-to-bigquery'
    google_cloud_options.staging_location = f"gs://{BUCKET_NAME}/stagging"
    google_cloud_options.temp_location = f"gs//{BUCKET_NAME}/temp"
    pipeline_options.view_as(beam.options.pipeline_options.StandardOptions).runner = 'DataflowRunner'
    google_cloud_options.region = REGION_NAME

    table_schema = {
        'fields': [
            {'name': 'type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'amount', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'oldbalanceOrg', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'newbalanceOrig', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'oldbalanceDest', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'newbalanceDest', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'isFraud', 'type': 'INTEGER', 'mode': 'NULLABLE'},
        ]
    }

    with beam.Pipeline(options=pipeline_options) as p:
        (p
         | 'ReadFromGCS' >> ReadFromText(f"gs://{BUCKET_NAME}/{FILENAME}", skip_header_lines=1)
         | 'ParseCSV' >> beam.ParDo(ParseCSV())
         | 'WriteToBigQuery' >> WriteToBigQuery(
             "loan_data.fraud_data",
             schema=table_schema,
             write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
         )
        
if __name__ == "__main__":
    main()