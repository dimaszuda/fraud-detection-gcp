import os
import joblib
import json
import pickle
import numpy as np
import pandas as pd
from time import time
import xgboost as xgb
from importlib import import_module
from sklearn import svm
from typing import List, Dict
from pandas import DataFrame
from collections import Counter
from dependecies import logging
from training.train import build_model, train_model
from models.schema import metrics_evaluation, algoritma_model
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
from kfp.dsl import (
    component,
    Output,
    Input,
    InputPath,
    Model,
    Metrics,
    Dataset,
    ClassificationMetrics
)

from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    f1_score
)
BASE_IMAGE = "gcr.io/deeplearning-platform-release/xgboost-cpu"

@component(base_image=BASE_IMAGE)
def calculate_metrics(
        metrics: List[str],
        algoritma: List[str],
        model: Output[Model], 
        training_data: InputPath(),
        test_data: InputPath(),
        evaluation: Output[Dataset]
) -> None:
    try:
        # Read dataset train and test from Big Query
        with open(training_data + ".pkl", 'rb') as file:
            train_data = pickle.load(file)
        X_train = train_data['x_train']
        y_train = train_data['y_train']

        with open(test_data + ".pkl", 'rb') as file:
            test_data = pickle.load(file)
        X_test = test_data['x_test']
        y_test = test_data['y_test']

        model_evaluation = build_model(metrics, algoritma)
        RF, svc, ada, xgb, gradient = train_model(X_train, y_train)
        model_dict = {
            'Random Forest': RF,
            'SVC': svc,
            'Ada Boost': ada,
            'XGBoost': xgb,
            'Gradient Boost': gradient
        }
        
        logging.info("Start evaluate models...")
        start_time = time()
        os.makedirs(model.path, exist_ok=True)
        for name, mod in model_dict.items():
            model_evaluation.loc[name, 'train_accuracy'] = accuracy_score(y_true=y_train, y_pred=mod.predict(X_train))
            model_evaluation.loc[name, 'train_precision'] = precision_score(y_true=y_train, y_pred=mod.predict(X_train))
            model_evaluation.loc[name, 'train_recall'] = recall_score(y_true=y_train, y_pred=mod.predict(X_train))
            model_evaluation.loc[name, 'train_f1'] = f1_score(y_true=y_train, y_pred=mod.predict(X_train))

            model_evaluation.loc[name, 'test_accuracy'] = accuracy_score(y_true=y_test, y_pred=mod.predict(X_test))
            model_evaluation.loc[name, 'test_precision'] = precision_score(y_true=y_test, y_pred=mod.predict(X_test))
            model_evaluation.loc[name, 'test_recall'] = recall_score(y_true=y_test, y_pred=mod.predict(X_test))
            model_evaluation.loc[name, 'test_f1'] = f1_score(y_true=y_test, y_pred=mod.predict(X_test))

            logging.info(f"Save model {name} to: {model.path}")
            joblib.dump(mod, model.path + f"/{name}.joblib")
        
        end_time = time()
        process_time = end_time - start_time
        logging.info(f"Process evaluate model: {process_time}")

        os.makedirs(model.path, exist_ok=True)
        # save evaluation metrics as csv file
        logging.info(f"Save evaluation metrics result to: {evaluation.path}")
        model_evaluation.to_csv(evaluation.path, sep=",", header=True, index=True)
        logging.info("Successfully evaluate models...")

    except Exception as E:
        logging.error(f"Error while evaluate model: {E}")

    return model

@component(base_image=BASE_IMAGE)
def evaluation_metrics(
        predictions: Input[Dataset],
        metrics_name: List[str],
        model_names: List[str],
        dict_keys: Dict[str, int],
        true_column_name: str,
        pred_column_name: str,
        metrics: Output[ClassificationMetrics],
        kpi: Output[Metrics],
        eval_metrics: Output[Metrics]
) -> None:
    
    for model_name in model_names:
        filename = f"{model_name}_predictions.csv"
        filepath = os.path.join(predictions.path, filename)
        results = pd.read_csv(filepath)

        results['class_true_clean'] = results[true_column_name].astype(str).map(dict_keys)
        results['class_pred_clean'] = results[pred_column_name].astype(str).map(dict_keys)

        module = import_module("sklearn.metrics")
        metrics_dict = {}
        for metrics in metrics_name:
            metric_func = getattr(module, metrics)
            if metrics == "f1_score":
                metric_val = metric_func(results['class_true_clean'], results['class_pred_clean'], average=None)
            else:
                metric_val = metric_func(results['class_true_clean'], results['class_pred_clean'])
            
            metric_val = np.round(np.average(metric_val), 4)
            metrics_dict[f"{metrics}"] = metric_val
            kpi.log_metric(f"{metrics}", metric_val)

            with open(kpi.path, "w") as f:
                json.dump(kpi.metadata, f)
            logging.info(f"{metrics}: {metric_val:.3f}")
        
        with open(eval_metrics.path, "w") as f:
            json.dump(metrics_dict, f)

        # To generate the confusion matrix plot
        confusion_matrix_func = getattr(module, "confusion_matrix")
        metrics.log_confusion_matrix(
            list(dict_keys.values()),
            confusion_matrix_func(results['class_true_clean'], results['class_pred_clean']).tolist()
        )
        
        # Dumping metrics metadata
        with open(metrics.path, "w") as f:
            json.dump(metrics.metadata, f)