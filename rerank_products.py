import pandas as pd
import joblib


def make_prediction(model, attributes_list, product_ids):
    predictions = model.predict(attributes_list)
    predicted_scores = {product_id: score for product_id, score in zip(product_ids, predictions)}
    reranked_product_ids = sorted(predicted_scores, key=predicted_scores.get, reverse=True)
    return reranked_product_ids, predicted_scores


class ProductPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.product_attributes = []
        self.product_ids = []

    def add_product(self, product_id, attributes):
        self.product_ids.append(product_id)
        self.product_attributes.append(attributes)

    def predict_reranked_product_ids(self):
        reranked_product_ids, predicted_scores = make_prediction(self.model, self.product_attributes, self.product_ids)
        return reranked_product_ids, predicted_scores


def main():
    model_path = 'model/search_reranking_model.joblib'
    predictor = ProductPredictor(model_path)

    # Add products with IDs and attributes
    product_1_id = 'SKU-0000-12345'
    product_1_attributes = [4.0, 0, 5.0, 1, 0.020829, 0.552162, 0.005076, 0.010152, 1.0, 0.149546,
                            0.046763, -0.423838, -0.000822, -0.097556, -0.787021]
    predictor.add_product(product_1_id, product_1_attributes)

    product_2_id = 'SKU-0001-54321'
    product_2_attributes = [3.5, 1, 4.5, 0, 0.012345, 0.432109, 0.003456, 0.009876, 0.8, 0.135791,
                            0.056789, -0.387654, -0.001234, -0.087654, -0.678901]
    predictor.add_product(product_2_id, product_2_attributes)

    # Predict reranked product IDs and scores
    reranked_product_ids, predicted_scores = predictor.predict_reranked_product_ids()

    # Save scores to CSV
    results_df = pd.DataFrame({'Product ID': reranked_product_ids, 'Score': [predicted_scores[id] for id in reranked_product_ids]})
    results_df.to_csv('out/reranked_product_scores.csv', index=False)

    print("Reranked Product IDs:", reranked_product_ids)
    print("Predicted Scores:")
    print(results_df)


if __name__ == '__main__':
    main()
