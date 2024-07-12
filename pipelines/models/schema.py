from google.cloud import bigquery
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

bq_schema = [
    bigquery.SchemaField("type", "STRING", "NULLABLE"),
    bigquery.SchemaField("amount", "FLOAT", "NULLABLE"),
    bigquery.SchemaField("oldbalanceOrg", "FLOAT", "NULLABLE"),
    bigquery.SchemaField("newbalanceOrig", "FLOAT", "NULLABLE"),
    bigquery.SchemaField("oldbalanceDest", "FLOAT", "NULLABLE"),
    bigquery.SchemaField("newbalanceDest", "FLOAT", "NULLABLE"),
    bigquery.SchemaField("isFraud", "INTEGER", "NULLABLE")
]

true_column = "class_true"
pred_column = "class_pred"

algoritma_model = ['Random Forest', 'SVC', 'Ada Boost', 'XGBoost', 'Gradient Boost']
metrics_evaluation = ['train_accuracy', 'train_precision', 'train_recall', 'train_f1',
                        'test_accuracy', 'test_precision', 'train_recall', 'train_f1']
metric_names = [
    "accuracy_score", 
    "recall_score", 
    "precision_score", 
    "f1_score"
]

