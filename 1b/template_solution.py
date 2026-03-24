# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def transform_features(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component in a given row of X)
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant feature: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: matrix of floats: dim = (700,21), transformed input with 21 features
    """
    #X_transformed = np.zeros((700, 21)) cHANGED TO
    X_transformed = np.zeros((X.shape[0], 21))
    # TODO: Enter your code here
    X_transformed[:, 0:5]  = X
    X_transformed[:, 5:10] = X ** 2
    X_transformed[:, 10:15] = np.exp(X)
    X_transformed[:, 15:20] = np.cos(X)
    X_transformed[:, 20]   = 1.0
    #TODO END

    #assert X_transformed.shape == (700, 21) CHANGED TO
    assert X_transformed.shape == (X.shape[0], 21)
    
    return X_transformed


def fit_logistic_regression(X, y):
    """
    This function receives training data points, transforms them, and then fits the logistic regression on this 
    transformed data. Finally, it outputs the weights of the fitted logistic regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of integers \in {0,1}, dim = (700,), input labels

    Returns
    ----------
    weights: array of floats: dim = (21,), optimal parameters of logistic regression
    """
    weights = np.zeros((21,))
    X_transformed = transform_features(X)
    # TODO: Enter your code here
    def nll(w):
        logits = X_transformed @ w
        return np.sum(np.logaddexp(0, logits) - y * logits)

    def nll_grad(w):
        return X_transformed.T @ (expit(X_transformed @ w) - y)

    best_loss, best_w = np.inf, np.zeros(21)
    rng = np.random.default_rng(42)
    for w0 in [np.zeros(21)] + [rng.normal(0, 0.1, 21) for _ in range(19)]:
        res = minimize(nll, w0, jac=nll_grad, method="L-BFGS-B",
                       options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10})
        if res.fun < best_loss:
            best_loss, best_w = res.fun, res.x

    weights = best_w
    #TODO END

    assert weights.shape == (21,)
    return weights


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function retrieving optimal LR parameters
    w = fit_logistic_regression(X, y)
    # Save results in the required format
    np.savetxt("./results.csv", w, fmt="%.12f")

    # ── Quick training accuracy ──────────────────────────────
    Phi = np.column_stack([X, X**2, np.exp(X), np.cos(X), np.ones(len(X))])
    probs = expit(Phi @ w)
    preds = (probs >= 0.5).astype(int)
    print(f"Training accuracy : {np.mean(preds == y)*100:.2f}%")

    # ── 5-fold CV ────────────────────────────────────────────
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []

    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        w_cv = fit_logistic_regression(X_tr, y_tr)
        Phi_val = np.column_stack([X_val, X_val**2, np.exp(X_val), np.cos(X_val), np.ones(len(X_val))])
        preds_val = (expit(Phi_val @ w_cv) >= 0.5).astype(int)
        cv_accs.append(np.mean(preds_val == y_val))

    print(f"5-fold CV accuracy : {np.mean(cv_accs)*100:.2f}% ± {np.std(cv_accs)*100:.2f}%")