"""
model_train.py
Author: Akhilesh K
Date: June 11, 2023
"""

import pandas as pd
import numpy as np
import datetime
import pytz
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import joblib


def read_csv_with_progress(filename, chunksize):
    cache_file = "data/cached_dataframe.joblib"
    try:
        df = joblib.load(cache_file)
        print("Cached DataFrame loaded successfully.")
        return df
    except FileNotFoundError:
        print("Cached DataFrame not found. Reading from CSV file.")
        df_list = []
        total_rows = sum(1 for _ in open(filename, "r", encoding="utf-8"))

        with tqdm(total=total_rows) as pbar:
            for chunk in pd.read_csv(filename, chunksize=chunksize):
                pbar.update(len(chunk))
                df_list.append(chunk)
        
        df = pd.concat(df_list, ignore_index=True)
        joblib.dump(df, cache_file)
        return df

def map_category_to_numeric(df, column, mapping):
    df[column] = df[column].str.lower().map(mapping)


def normalize_column(df, column):
    df[column + "_normalized"] = (df[column] - df[column].min()) / (
        df[column].max() - df[column].min()
    )


def scale_column(df, column):
    scaler = StandardScaler()
    df[column + "_standardized"] = scaler.fit_transform(df[[column]])


def calculate_target(df, ctr_weight, position_weight):
    df["target"] = (df["ctr_normalized"] * ctr_weight) + (df["position"] * position_weight)
    return df


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Shuffle
    df_sample = df.sample(frac=1, random_state=12)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    if target_variable is not None:
      grouped_df = df_sample.groupby(target_variable)
      arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

      train_ds = pd.concat([t[0] for t in arr_list])
      val_ds = pd.concat([t[1] for t in arr_list])
      test_ds = pd.concat([v[2] for v in arr_list])

    else:
      indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
      train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds, test_ds


class AttributePredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.model.predict(X)


def main():
    filename = "bgq_data_search.csv"
    chunksize = 1000000

    df = read_csv_with_progress(filename, chunksize)

    category_mapping = {
        'gold': 3,
        'silver': 2,
        'diamond': 4,
        'bronze': 1
    }
    print(f"Mapping Seller badges..")
    map_category_to_numeric(df, 'seller_badge', category_mapping)

    df['official_store'] = df['official_store'].astype(bool).astype(int)

    print(f"Normalizing columns..")
    normalize_columns = ['lifetimeorder_count', 'product_score', 'cvrate', 'a2crate', 'ctr', 'search_result_score']
    for column in normalize_columns:
        normalize_column(df, column)

    scaler = MinMaxScaler()
    normalize_column(df, 'price')
    df['price_scaled'] = scaler.fit_transform(df['price'].values.reshape(-1, 1))

    df['price_scaled_log'] = np.log(df['price'] + 1)

    print(f"Dropping unused columns..")
    df.drop(['pinpoint', 'sales_velocity_components', 'product_tag'], axis=1, inplace=True)

    df['product_created_date'] = pd.to_datetime(df['product_created_date'], errors='coerce')

    print(f"Setting Product Age..")
    current_day = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('utc'))
    df['product_age'] = (current_day - df['product_created_date']).dt.days

    scaler = MinMaxScaler()
    df['product_age_normalized'] = scaler.fit_transform(df[['product_age']])

    scaler = StandardScaler()
    df['product_age_standardized'] = scaler.fit_transform(df[['product_age']])

    scale_columns = ['price', 'ctr_price', 'atoc_price']
    for column in scale_columns:
        scale_column(df, column)

    df = calculate_target(df, ctr_weight=0.9, position_weight=0.2)
    print(f"Setting dataframe for model training..")
    selected_features = ['seller_badge', 'official_store', 'product_rating', 'a2crclicks',
                         'lifetimeorder_count_normalized', 'product_score_normalized',
                         'cvrate_normalized', 'a2crate_normalized', 'ctr_normalized',
                         'search_result_score_normalized', 'product_age_normalized', 'product_age_standardized',
                         'price_standardized', 'target', 'ctr_price_standardized',
                         'atoc_price_standardized']
    numeric_df = df[selected_features].copy()

    # Drop rows containing missing or infinite values
    numeric_df_clean = numeric_df.dropna()
    train_split = 0.8  # Percentage of data for training set
    val_split = 0.1  # Percentage of data for validation set
    test_split = 0.1  # Percentage of data for test set
    train_ds, val_ds, test_ds = get_dataset_partitions_pd(numeric_df_clean, train_split, val_split, test_split, target_variable='target')

    print("Train set shape:", train_ds.shape)
    print("Validation set shape:", val_ds.shape)
    print("Test set shape:", test_ds.shape)

    X_train = train_ds.drop('target', axis=1)
    y_train = train_ds['target']

    X_val = val_ds.drop('target', axis=1)
    y_val = val_ds['target']

    X_test = test_ds.drop('target', axis=1)
    y_test = test_ds['target']

    print(f"Training Search Reranking model...")
    search_linear_model = LinearRegression()
    search_linear_model.fit(X_train, y_train)
    print(f"Training Completed, analyzing the trained model..")

    y_val_pred = search_linear_model.predict(X_val)
    y_test_pred = search_linear_model.predict(X_test)

    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Validation Set MSE:", val_mse)
    print("Validation Set R-squared:", val_r2)
    print("Test Set MSE:", test_mse)
    print("Test Set R-squared:", test_r2)

    joblib.dump(search_linear_model, "model/search_reranking_model.joblib")
    print(f"Saving the trained model as search_reranking_model.joblib")

if __name__ == "__main__":
    main()
