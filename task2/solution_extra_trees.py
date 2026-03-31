import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data():
    """
    Load train/test CSVs, drop rows with missing targets, and build feature matrices.

    Returns
    -------
    X_train : pd.DataFrame
        Training features.
    y_train : np.ndarray
        Observed Swiss prices.
    X_test : pd.DataFrame
        Test features.
    """
     # Load test data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables  

    # Keep only rows with observed target values.
    train_df = train_df.dropna(subset=["price_CHF"]).reset_index(drop=True)

    X_train = train_df.drop(columns=["price_CHF"])
    y_train = train_df["price_CHF"].to_numpy()
    X_test = test_df.copy()

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    assert X_train.shape[1] == X_test.shape[1], "Train/test feature mismatch"
    assert X_train.shape[0] == y_train.shape[0], "Invalid training shapes"
    return X_train, y_train, X_test


class Model:
    def __init__(self):
        numeric_features = [
            "price_AUS", "price_CZE", "price_GER", "price_ESP",
            "price_FRA", "price_UK", "price_ITA", "price_POL", "price_SVK"
        ]
        categorical_features = ["season"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    SimpleImputer(strategy="median"),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        regressor = ExtraTreesRegressor(
            n_estimators=1000,
            random_state=42,
            n_jobs=-1,
        )

        self.model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("regressor", regressor),
            ]
        )

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        y_pred = self.model.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],), "Invalid prediction shape"
        return y_pred


if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = Model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    submission = pd.DataFrame({"price_CHF": y_pred})
    submission.to_csv("results.csv", index=False)
    print("Saved results.csv")
