# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, DotProduct, ConstantKernel,
)
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


NUMERIC_COLS = [
    "price_AUS", "price_CZE", "price_GER", "price_ESP",
    "price_FRA", "price_UK", "price_ITA", "price_POL", "price_SVK",
]
SEASON_COL = "season"
 
# Different kernels to test
"""KERNELS = {
    "RationalQuadratic + White": RationalQuadratic(length_scale=1.0, alpha=1.0)
                                 + WhiteKernel(noise_level=0.1),
    "RBF + White":               ConstantKernel(1.0) * RBF(length_scale=1.0)
                                 + WhiteKernel(noise_level=0.1),
    "Matern(1.5) + White":       ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
                                 + WhiteKernel(noise_level=0.1),
    "Matern(2.5) + White":       ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                                 + WhiteKernel(noise_level=0.1),
    "DotProduct + White":        DotProduct(sigma_0=1.0)
                                 + WhiteKernel(noise_level=0.1),
}"""

def load_data():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    train_df = train_df.dropna(subset=["price_CHF"]).reset_index(drop=True)
 
    y_train = train_df["price_CHF"].to_numpy(dtype=float)
 
    # One-hot encode the season column for both train and test tgt => both matrices get the same set of dummy columns no matter the season
    season_dummies = pd.get_dummies(
        pd.concat([train_df[SEASON_COL], test_df[SEASON_COL]], axis=0),
        prefix="season",
        drop_first=False,
    ).astype(float)
    n_train = len(train_df)
    season_train = season_dummies.iloc[:n_train].to_numpy()
    season_test  = season_dummies.iloc[n_train:].to_numpy()
 
    # Impute with correlations instead of medians
    imputer = IterativeImputer(
        estimator=GradientBoostingRegressor(n_estimators=50, random_state=0),
        max_iter=10,
        random_state=42,
    )
 
    train_numeric = train_df[NUMERIC_COLS].to_numpy(dtype=float)
    test_numeric  = test_df[NUMERIC_COLS].to_numpy(dtype=float)
 
    imputer.fit(train_numeric)
    train_numeric_imputed = imputer.transform(train_numeric)
    test_numeric_imputed  = imputer.transform(test_numeric)
 
    # GP kernels are distance-based and sensitive to scaling => StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_numeric_imputed)
    train_numeric_scaled = scaler.transform(train_numeric_imputed)
    test_numeric_scaled  = scaler.transform(test_numeric_imputed)
 
    # Concatenate numeric and season features
    X_train = np.hstack([train_numeric_scaled, season_train])
    X_test  = np.hstack([test_numeric_scaled,  season_test])


    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test


class Model(object):
    def __init__(self):
        super().__init__()
        self._x_train = None
        self._y_train = None

        self._model = None

    def _use_gauss(self, kernel) -> GaussianProcessRegressor:
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,  # multiple restarts avoid local optima in kernel fitting
            normalize_y=True,        # centres the GP prior on the data mean
            random_state=42,
            alpha=1e-6,              # small jitter for numerical stability
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        #TODO: Define the model and fit it using (X_train, y_train)
        self._x_train = X_train
        self._y_train = y_train

        """
        # Select kernel with 5-fold CV
        print("Selecting kernel via 5-fold CV ...")
        best_score, best_name, best_kernel = -np.inf, None, None
 
        for name, kernel in KERNELS.items():
            gpr = self._use_gauss(kernel)
            scores = cross_val_score(gpr, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
            mean_r2 = scores.mean()
            print(f"  {name:35s}  CV R² = {mean_r2:.4f} ± {scores.std():.4f}")
            if mean_r2 > best_score:
                best_score, best_name, best_kernel = mean_r2, name, kernel
 
        print(f"\nSelected kernel: {best_name}  (CV R² = {best_score:.4f})")
 
        # Retrain on the full training set with the winning kernel
        self._model = self._use_gauss(best_kernel)
        """

        self._model = self._use_gauss(ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1))
        self._model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        y_pred=np.zeros(X_test.shape[0])
        #TODO: Use the model to make predictions y_pred using test data X_test
        y_pred = self._model.predict(X_test)
        
        assert y_pred.shape == (X_test.shape[0],), "Invalid data shape"
        return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = load_data()
    model = Model()
    # Use this function to fit the model
    model.fit(X_train=X_train, y_train=y_train)
    # Use this function for inference
    y_pred = model.predict(X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
