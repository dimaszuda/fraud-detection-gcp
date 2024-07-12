import os
import joblib
import pickle
import pandas as pd
from time import time
import xgboost as xgb
from sklearn import svm
from typing import List
from pandas import DataFrame
from collections import Counter
from dependecies import logging
from models.schema import metrics_evaluation, algoritma_model
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier
)
from kfp.dsl import (
    Output,
    InputPath,
    Model,
    Dataset,
    component
)

BASE_IMAGE = "gcr.io/deeplearning-platform-release/xgboost-cpu"

def build_model(metrics: List[str], model: List[str]) -> DataFrame:
    models = pd.DataFrame(
        index=metrics,
        columns=model
    )
    return model

@component(base_image=BASE_IMAGE)
def train_model(
        training_data: InputPath(),
        model: Output[Model],
        n_estimator: int = 100,
        max_depth: int = 1, 
        random_state: int = 55, 
        learning_rate: int = 0.05,
        n_jobs: int  = -1,
        use_label_encoder: bool = False,
        eval_metrics: str = "logloss",
        kernel: str = "linear",
        C: float = 1.0,
):
    try:
        logging.info(f"Get dataset from Big Query")
        with open(training_data + ".pkl", 'rb') as file:
            train_data = pickle.load(file)
        X_train = train_data['x_train']
        y_train = train_data['y_train']

        logging.info("Training models...")
        start_time = time()
        """
        Training Random Forest Model
        """
        logging.info("Training Random Forest Model")
        rf_start = time()
        RF = RandomForestClassifier(
            n_estimators=n_estimator,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
        RF.fit(X_train, y_train)
        rf_end = time()
        process_time = rf_end - rf_start
        logging.info(f"Process training Random Forest: {process_time}")

        """
        Training Support Vector Machine with Support Vector Classifier.
        params:
            kernel = "linear" 
            (it can be replace with other kernel like 'poly', 'rbf', 'sigmoid', 'precomputed')
        """
        logging.info("Training Support Vector Machine Model")
        svc_start = time()
        svc = svm.SVC(
            kernel=kernel,
            C=C
        )
        svc.fit(X_train, y_train)
        svc_end = time()
        process_time = svc_end - svc_start
        logging.info(f"process training SVC: {process_time}")

        """
        Training Ada Boost Classifier Model with learning rate 0.05
        """
        ada_start = time()
        ada = AdaBoostClassifier(
            learning_rate=learning_rate,
            random_state=random_state
        )
        ada.fit(X_train, y_train)
        ada_end = time()
        process_time = ada_end - ada_start
        logging.info(f"Process training Ada: {process_time}")

        """
        Training XG Boost Model with XG Classifier
        params:
            scale_post_weight -> help address class imbalances in a dataset
            use_label_encoder = False ->  we set to False because we have already encoded the dataset
        """
        xg_start = time()
        scale_pos_weight = Counter(y_train)[0] / Counter(y_train)[1]
        xgb = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, 
            use_label_encoder=use_label_encoder, 
            eval_metrics=eval_metrics, 
            random_state=random_state
        )
        xgb.fit(X_train, y_train)
        xg_end = time()
        process_time = xg_end - xg_start
        logging.info(f"Process training XGBoost: {process_time}")

        """
        Training Gradient Boosting Model
        """
        gradient_start = time()
        gradient = GradientBoostingClassifier(
            n_estimators=n_estimator,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        gradient.fit(X_train, y_train)
        gradient_end = time()
        process_time = gradient_end - gradient_start
        logging.info(f"Process time Gradient Boosting: {process_time}")

        end_time = time()
        logging.info("Successfully training models...")
        logging.info(f"Total process training 5 models: {end_time - start_time}")

    except Exception as E:
        logging.error(f"Error while training models: {E}")

    return RF, svc, ada, xgb, gradient



