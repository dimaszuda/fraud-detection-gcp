import os
import json
import pickle
import joblib
import pandas as pd
from typing import List
from dependecies import logging
from kfp.dsl import (
    Input,
    InputPath,
    Model,
    Dataset,
    Output,
    component
)
BASE_IMAGE = "gcr.io/deeplearning-platform-release/xgboost-cpu"

@component(base_image=BASE_IMAGE)
def test_model(
    model_names: List[str],
    test_data: InputPath(),
    model: Input[Model],
    predictions: Output[Dataset]
) -> None:
    
    with open(test_data + ".pkl", "rb") as file:
        test_data = pickle.load(file)
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    for model_name in model_names:
        model_path = os.path.join(model.path, f"{model_name}.joblib")
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        df = pd.DataFrame(
            {
                'class_true': y_test.tolist(),
                'class_pred': y_pred.tolist()
            }
        )

        output_filename = f"{model_name}_prediction.csv"
        output_path = os.path.join(predictions.path, output_filename)
        df.to_csv(output_path, sep=",", header=True, index=False)