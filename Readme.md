# Search Reranking Model - POC

This repository contains the source code for a search reranking model that is designed to improve the accuracy and relevance of search results for e-commerce product searches. The model is based on linear regression and uses various features of the search query results and product data to rerank the search results.

 It takes in a list of products with their attributes and returns the reordered product IDs and their corresponding predicted scores.

## File Structure

The directory structure of the repository is as follows:

```
search_reranking_model/
├── Readme.md
├── data/
│   ├── bgq_data_search.csv
│   └── cached_dataframe.joblib
├── docs/
│   └── reranking-model-arch.png
├── model/
│   └── search_reranking_model.joblib
├── model_train.py
├── notebooks/
│   ├── Workbook-assumptions_of_LR.ipynb
│   └── search_reranking_model.ipynb
├── out/
│   └── reranked_product_scores.csv
├── requirements.txt
├── rerank_products.py
└── reranking_model.py
```

* The `data` directory contains the input data for the model, including the search query data (`bgq_data_search.csv`) and a cached version of the preprocessed data (`cached_dataframe.joblib`). The `model` directory contains the trained model (`search_reranking_model.joblib`).

* The `docs` directory contains documentation related to the model architecture, including a diagram (`reranking-model-arch.png`) that illustrates the various components and inputs.

* The `model_train.py` file contains the code to train the model on the input data.

* The `notebooks` directory contains Jupyter notebooks that were used for exploratory data analysis and model testing. The `Workbook-assumptions_of_LR.ipynb` notebook explores the assumptions of logistic regression and the `search_reranking_model.ipynb` notebook demonstrates the use of the trained model on new data.

* The `out` directory contains the final output of the model, which is a csv file (`reranked_product_scores.csv`) containing the reranked scores for each product based on the search query.

* The `requirements.txt` file lists the required Python packages and their versions.

* The `rerank_products.py` script is the main script for running the model on new data. It preprocesses the data, applies the model, and generates the reranked product scores.

* The `reranking_model.py` file contains the code for defining and applying the logistic regression model.

## Installation

* Clone the repository:
`git clone https://github.com/akhilesh-k/search-reranking-model.git`
* Install the required packages:
`pip install -r requirements.txt`

## Usage

To use the product predictor, follow these steps:

* Setup training data, if you want to play with the model. You can use the bgq query to import the dataframe if you have the access.

* Prepare your input data as a list of products with their IDs and attributes.
* Instantiate the ProductPredictor class by passing the path to the trained machine learning model:

    ```
    from product_predictor import ProductPredictor
    model_path = 'path/to/model.joblib'
    predictor = ProductPredictor(model_path)
    ```

* Add each product to the ProductPredictor instance using the `add_product` method:
   `product_id = 'SKU-0000-12345'`
   `attributes = [4.0, 0, 5.0, 1, 0.020829, 0.552162, 0.005076, 0.010152, 1.0, 0.149546,
                 0.046763, -0.423838, -0.000822, -0.097556, -0.787021]`
   `predictor.add_product(product_id, attributes)`
* Call the predict_reranked_product_ids method on the ProductPredictor instance to get the predicted reordered product IDs and their corresponding scores:

    `reranked_product_ids, predicted_scores = predictor.predict_reranked_product_ids()`

* Save the predicted scores to a CSV file:
   `results_df = pd.DataFrame({'Product ID': reranked_product_ids, 'Score': [predicted_scores[id] for id in reranked_product_ids]})`
   `results_df.to_csv('path/to/output.csv', index=False)`

## License

This project is licensed under the MIT License. See the LICENSE file for details.