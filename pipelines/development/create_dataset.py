import pickle
from google.cloud import bigquery
import pandas as pd
from typing import List, NamedTuple
from dependecies import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from kfp.dsl import (
    component,
    Input,
    Dataset,
    OutputPath,
)

BASE_IMAGE = "gcr.io/deeplearning-platform-release/xgboost-cpu"

@component(base_image=BASE_IMAGE)
def create_df(
        data_input: Input[Dataset],
        dataset_train: OutputPath(),
        dataset_test: OutputPath()
) -> NamedTuple("Outputs", [("dict_keys", dict), ("shape_train", int), ("shape_test", int)]):
    """
    function to generate pandas dataframe from Big Query. This dataframe
    is aimed to inference Model Training
    """
    logging.info("Extract dataset from Big Query")
    try:
        df = pd.read_csv(data_input.path)
        df.dropna()
        df.drop_duplicates()
        df = pd.concat(
            [
                df,
                pd.get_dummies(
                    df['type']
                ).astype(int)
            ],
            axis=1
        )

        df.drop(
            ['type'], 
            axis=1, 
            inplace=True
        )

        if len(df) != 0:
            logging.info("Drop label and attributes")
            X = df.drop(columns=['isFraud'], axis=1)
            y = df['isFraud']
            dic_keys = {k: label for k, label in enumerate(sorted(y.unique()))}

            logging.info("Create Undersampling of the dataset")
            """
            We will do undersampling because dataset is imbalanced,
            and will causing overfitting.
            label fraud amount is 6M+
            label not fraud is just 8K
            """
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit(X, y)

            logging.info("Split dataset into train and test set")
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=123)

            logging.info("Normalize dataset")
            """
            Each column that doesn't have a value between 0 and 1 will be normalized
            """
            numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            scaler = StandardScaler()
            scaler.fit(X_train[numerical_features])
            X_train[numerical_features] = scaler.transform(
                                            X_train.loc[:, numerical_features]
                                        )
            X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
            x_train_results = {'X_train': X_train, 'y_train': y_train}
            x_test_results = {'X_test': X_test, 'y_test': y_test}

            logging.info("Create pickle file and send to Big Query")
            with open(dataset_train + ".pkl", "wb") as file:
                pickle.dump(x_train_results, file)
            with open(dataset_test + ".pkl", "wb") as file:
                pickle.dump(x_test_results, file)
            logging.info(f"[END] - CREATE SETS, dataset was split")

            return (dic_keys, len(X_train), len(X_test))
        
        else:
            logging.error(f"Cannot create dataset because dataset is empty")
            return None, None
        
    except Exception as E:
        logging.error(f"Unexpected Error: {E}")





