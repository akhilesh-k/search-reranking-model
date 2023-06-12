"""
reranking_model.py
Author: Akhilesh K
Date: June 11, 2023
"""

import pandas as pd
from tqdm import tqdm
import datetime
import pytz
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def read_csv_with_progress(filename, chunksize):
    total_rows = sum(1 for _ in open(filename, 'r', encoding='utf-8'))
    df_list = []
    with tqdm(total=total_rows) as pbar:
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            pbar.update(len(chunk))
            df_list.append(chunk)
    return pd.concat(df_list, ignore_index=True)


def map_category_to_numeric(df, column_name, category_mapping):
    df[column_name] = df[column_name].str.lower().map(category_mapping)


def normalize_column(df, column_name):
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    df[f'normalized_{column_name}'] = (df[column_name] - min_val) / (max_val - min_val)


def scale_column(df, column_name):
    scaler = StandardScaler()
    column_data = df[column_name].values.reshape(-1, 1)
    df[f'{column_name}_standardized'] = scaler.fit_transform(column_data)


def calculate_target(df, ctr_weight, position_weight):
    df['target'] = np.where(df['normalized_ctr'].notnull() & df['position'].notnull(),
                            (df['normalized_ctr'] * ctr_weight + np.minimum(df['position'], 40) * position_weight) /
                            (ctr_weight + position_weight),
                            np.nan)
    return df


def get_dataset_partitions(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
    assert (train_split + test_split + val_split) == 1
    assert val_split == test_split

    df_sample = df.sample(frac=1, random_state=12)

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


def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, r2


def make_prediction(model, product_attributes):
    input_data = pd.DataFrame(
        [product_attributes],
        columns=[
            "position", "seller_badge", "official_store", "product_rating", "a2crclicks",
            "normalized_lifetimeorder_count", "normalized_product_score", "normalized_cvr",
            "normalized_a2crate", "normalized_ctr", "normalized_search_result_score",
            "product_age_normalized", "product_age_standardized", "price_standardized",
            "ctr_price_standardized", "atoc_price_standardized"
        ],
    )
    return model.predict(input_data)


def main():
    filename = 'bgq_data_search.csv'
    chunksize = 1000000

    df = read_csv_with_progress(filename, chunksize)

    category_mapping = {
        'gold': 3,
        'silver': 2,
        'diamond': 4,
        'bronze': 1
    }
    map_category_to_numeric(df, 'seller_badge', category_mapping)

    df['official_store'] = df['official_store'].astype(bool).astype(int)

    normalize_column(df, 'lifetimeorder_count')
    normalize_column(df, 'product_score')
    normalize_column(df, 'cvrate')
    normalize_column(df, 'a2crate')
    normalize_column(df, 'ctr')
    normalize_column(df, 'search_result_score')

    scaler = MinMaxScaler()
    normalize_column(df, 'price')
    df['price_scaled'] = scaler.fit_transform(df['price'].values.reshape(-1, 1))

    df['price_scaled_log'] = np.log(df['price'] + 1)

    df.drop(['pinpoint', 'sales_velocity_components', 'product_tag'], axis=1, inplace=True)

    df['product_created_date'] = pd.to_datetime(df['product_created_date'], errors='coerce')

    current_day = datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('utc'))
    df['product_age'] = (current_day - df['product_created_date']).dt.days

    scaler = MinMaxScaler()
    df['product_age_normalized'] = scaler.fit_transform(df[['product_age']])

    scaler = StandardScaler()
    df['product_age_standardized'] = scaler.fit_transform(df[['product_age']])

    scale_column(df, 'price')
    scale_column(df, 'ctr_price')
    scale_column(df, 'atoc_price')

    df = calculate_target(df, ctr_weight=0.7, position_weight=0.3)

    selected_features = ['position', 'seller_badge', 'official_store', 'product_rating', 'a2crclicks',
                         'normalized_lifetimeorder_count', 'normalized_product_score',
                         'normalized_cvr', 'normalized_a2crate', 'normalized_ctr',
                         'normalized_search_result_score', 'product_age_normalized', 'product_age_standardized',
                         'price_standardized', 'target', 'ctr_price_standardized',
                         'atoc_price_standardized']
    numeric_df = df[selected_features].copy()

    correlation = numeric_df.corr()
    print(correlation["target"].sort_values(ascending=False) * 100)

    numeric_df_clean = numeric_df.replace([np.inf, -np.inf], np.nan).dropna()

    X_new = sm.add_constant(numeric_df_clean)

    vif = [variance_inflation_factor(X_new.values, i) for i in range(X_new.shape[1])]
    print(vif)

    train_ds, val_ds, test_ds = get_dataset_partitions(numeric_df_clean, train_split=0.8, val_split=0.1, test_split=0.1,
                                                       target_variable='target')

    print("Train set shape:", train_ds.shape)
    print("Validation set shape:", val_ds.shape)
    print("Test set shape:", test_ds.shape)

    X_train = train_ds.drop('target', axis=1)
    y_train = train_ds['target']

    X_val = val_ds.drop('target', axis=1)
    y_val = val_ds['target']

    X_test = test_ds.drop('target', axis=1)
    y_test = test_ds['target']

    search_linear_model = LinearRegression()
    search_linear_model.fit(X_train, y_train)

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

    product_attributes = [73, 4.0, 0, 5.0, 1, 0.020829, 0.552162, 0.005076, 0.010152, 1.0, 0.149546,
                          0.046763, -0.423838, -0.000822, -0.097556, -0.787021]
    prediction = make_prediction(search_linear_model, product_attributes)
    print(prediction)


if __name__ == '__main__':
    main()
