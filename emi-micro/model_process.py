import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import datetime

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from prefect import task, flow



mlflow.set_tracking_uri('sqlite:///mlflow/runs_info.db')
mlflow.set_experiment('emotions-classifier')


train_df = pd.read_parquet('data/parquet/train_data.parquet')
test_df = pd.read_parquet('data/parquet/test_data.parquet')
val_df = pd.read_parquet('data/parquet/val_data.parquet')


@task(name="Loading and Splitting Dataset")
def load_split_dataset(path: str):

    print(f'Loading Dataset from {path} \n')
    data = pd.read_parquet(path=path)
    X = data["words"]
    y = data["emotions"]
    print(f'Dataset Loaded from {path} \n')

    return X, y


@task(name="Model Training in Progress")
def model_training(train_X, train_y):

    model_name = "Histogram Gradient Boosting Classifier"

    model_hyperparameters = {
        "loss" : "log_loss",
        "learning_rate" : 0.05,
        "min_samples_leaf" : 30
    }


    clf = HistGradientBoostingClassifier(
        loss=model_hyperparameters["loss"],
        learning_rate=model_hyperparameters["learning_rate"],
        min_samples_leaf=model_hyperparameters["min_samples_leaf"]
    )

    print(f'Training Model on Data \n')

    # coverting series object to list of arrays for Sklearn to trai easily 
    train_X_vectors = [x for x in train_X]

    clf.fit(train_X_vectors, train_y)

    print(f'Model Training Completed \n')

    return model_hyperparameters, clf, model_name


@task(name="Model Validation in Progress")
def model_validation(clf, val_X, val_y):

    val_X_vectors = [x for x in val_X]
    y_true = val_y

    print(f'Validation in progress \n')

    y_pred = clf.predict(val_X_vectors)

    print(f'Computing Metrics \n')

    val_f1_score = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    val_precision_score = precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
    val_recall_score = recall_score(y_true=y_true, y_pred=y_pred, average="weighted")

    print(f'Metrics Computed \n')

    scores = {
        "F1 Score" : val_f1_score,
        "Precision Score" : val_precision_score,
        "Recall Score" : val_recall_score
    }

    return scores


@task(name="Logging Metrics in Progress")
def logging(model_hyperparameters, metrics, model, model_name):

    print(f'Logging in Progress \n')

    with mlflow.start_run():
    
        mlflow.log_params(model_hyperparameters)

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="mlflow/artifacts",
            registered_model_name=f"{model_name} - {datetime.datetime.now()}"
        )

    print(f'Logging Completed \n')


@flow(name="Running Processes from Main Function")
def main(train_path, val_path):

    train_X, train_y = load_split_dataset(train_path)
    val_X, val_y = load_split_dataset(val_path)

    model_hyperparameters, clf, model_name = model_training(train_X=train_X, train_y=train_y)
    metrics = model_validation(clf, val_X=val_X, val_y=val_y)

    # log values to mlflow
    logging(
            model_hyperparameters=model_hyperparameters, 
            metrics=metrics, 
            model=clf, 
            model_name=model_name
    ) 

    return metrics


if __name__ == "__main__":
    train_path = 'data/parquet/train_data.parquet'
    val_path = 'data/parquet/val_data.parquet'

    model_metrics = main(train_path=train_path, val_path=val_path)

    print(model_metrics)