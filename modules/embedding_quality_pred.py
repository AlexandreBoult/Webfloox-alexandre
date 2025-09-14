import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class EmbeddingQualityPredictor:
    def __init__(self, model=None):
        # Default to RandomForest if no model is passed
        self.model = model if model else RandomForestRegressor(random_state=0)

    def train(self, X, y, use_grid_search=False):
        """
        Train the model on embeddings and grades.
        Optionally performs hyperparameter tuning with cross-validation.
        :param X: np.ndarray or pd.DataFrame, shape (n_samples, embedding_dim)
        :param y: list or np.ndarray, shape (n_samples,)
        :param use_grid_search: bool, whether to perform hyperparameter tuning
        """
        if use_grid_search:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=0),
                param_grid,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            print("Best parameters:", grid_search.best_params_)
        else:
            self.model.fit(X, y)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on a test set.
        Returns RMSE and R² score.
        """
        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        return rmse, r2

    def predict(self, embedding):
        """
        Predict quality grade for a single embedding.
        :param embedding: np.ndarray, shape (embedding_dim,)
        :return: float, predicted grade (between 1 and 5)
        """
        return self.model.predict([embedding])[0]

    def save(self, path="embedding_quality_model.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="embedding_quality_model.pkl"):
        self.model = joblib.load(path)


if __name__ == "__main__":
    # Example usage
    # Suppose you have a CSV with columns: embedding (as list) and grade
    # df = pd.read_csv("embeddings.csv")

    # Assume embeddings are stored as strings like "[0.1, 0.2, ...]"
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))

    X = np.vstack(df["embedding"].values)
    y = df["grade"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    predictor = EmbeddingQualityPredictor()
    predictor.train(X_train, y_train, use_grid_search=True)  # Enable tuning

    rmse, r2 = predictor.evaluate(X_test, y_test)
    print(f"Test RMSE: {rmse:.3f}, R²: {r2:.3f}")

    # Example prediction on a new embedding
    new_embedding = X_test[0]
    print("Predicted quality:", predictor.predict(new_embedding))
