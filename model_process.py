import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


train_df = pd.read_parquet('data/parquet/train_data.parquet')
test_df = pd.read_parquet('data/parquet/test_data.parquet')
val_df = pd.read_parquet('data/parquet/val_data.parquet')


def load_split_dataset(path):
    data = pd.read_parquet(path=path)
