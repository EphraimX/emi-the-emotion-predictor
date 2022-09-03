import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


train_df = pd.read_parquet('data/parquet/train_data.parquet')
test_df = pd.read_parquet('data/parquet/test_data.parquet')
val_df = pd.read_parquet('data/parquet/val_data.parquet')


def load_split_dataset(path):

    data = pd.read_parquet(path=path)
    X = data["words"]
    y = data["emotions"]

    return X, y


def model_training(clf, train_X, train_y):

    model_hyperparameters = {

    }

    clf = clf()

    clf.fit(train_X, train_y)

    return model_hyperparameters, clf   


def model_validation(clf, val_X, val_y):

    y_true = val_y
    y_pred = clf.predict(val_X)

    val_f1_score = f1_score(y_true=y_true, y_pred=y_pred)
    val_precision_score = precision_score(y_true=y_true, y_pred=y_pred)
    val_recall_score = recall_score(y_true=y_true, y_pred=y_pred)

    scores = {
        "F1 Score" : val_f1_score,
        "Precision Score" : val_precision_score,
        "Recall Score" : val_recall_score
    }

    return scores


def logging(model_hyperparameters, metrics):
    ...


def main(train_path, val_path):

    train_X, train_y = load_split_dataset(train_path)
    val_X, val_y = load_split_dataset(val_path)

    model_hyperparameters, clf = model_training(train_X=train_X, train_y=train_y)
    metrics = model_validation(clf, val_X=val_X, val_y=val_y)

    # log values to mlflow
    logging(model_hyperparameters=model_hyperparameters, metrics=metrics) 


if __name__ == "__main__":
    train_path = ''
    val_path = ''
    main(train_path=train_path, val_path=val_path)