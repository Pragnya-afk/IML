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
    #Initializes the empty result matrix:
    X_transformed = np.zeros((X.shape[0], 21))      # 700 is now dynamic (because it didn't fit)
    
    # TODO: Enter your code here
    # Sets the columns to the processed vectors (since the combined i-th component of every row is column i):
    X_transformed[:, 0:5]  = X              # 5 linear features
    X_transformed[:, 5:10] = X ** 2         # 5 quadratic features
    X_transformed[:, 10:15] = np.exp(X)     # 5 exponential features
    X_transformed[:, 15:20] = np.cos(X)     # 5 cosine features
    X_transformed[:, 20]   = 1.0            # 1 constant feature

    # Make sure the shape of the matrix hasn't been changed:
    assert X_transformed.shape == (X.shape[0], 21)      # 700 now dynamic
    return X_transformed                                # Return X with the features applied


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
    # Preparation (initializes the weights vector and transforms X):
    weights = np.zeros((21,))
    X_transformed = transform_features(X)
    
    # TODO: Enter your code here
    # Compute loss with log(1-sigm(pred)) = log(1+e^{pred})-yz
    def negativeLL(w):
        predicted = X_transformed @ w           # Matrix multiplication (@) of transformed X with weights for model prediction
        logLL = np.logaddexp(0, predicted)      # Is log(1+e^{pred})
        loss = logLL - y * predicted            # Vector of loss at each point
        return np.sum(loss)                     # Total loss
    # Compute the gradient of the loss
    def negativeLL_gradient(w):
        predicted = X_transformed @ w           # Same as above
        sigmoid = expit(predicted)              # same as \sigma(Xw)
        return X_transformed.T @ (sigmoid - y)  # Returns the gradient vector as the transformed matrix (transposed) times the difference until prediction is accurate

    best_loss, best_weights = np.inf, np.zeros(21)
    rng = np.random.default_rng(42)             # Generates a random number with 42 as the seed (to get the same results when running)
    for w0 in [np.zeros(21)] + [rng.normal(0, 0.1, 21) for _ in range(19)]:     # Since solution dependant on start: Creates starting points in in 0 and then 19 random vectors (because 21 weights w/ mean 0 and std 0.1 => 1 zero start, 19 random)
        res = minimize(                 # Tries to minimize negative log likelihood
                    negativeLL,                     # Loss function being minimized
                    w0,                             # Starting point (initial weights => each loop tries different one)
                    jac=negativeLL_gradient,        # Gradient of loss function
                    method="L-BFGS-B",              # Optimization algorithm
                    options={
                        "maxiter": 10000,   # Maximum # of iterations
                        "ftol": 1e-15,      # Stop when loss barely changes
                        "gtol": 1e-10       # Stop when gradient very small
                    }
        )
        # Keep the best result:
        if res.fun < best_loss:
            best_loss, best_weights = res.fun, res.x

    # Update weights tot he best result:
    weights = best_weights

    # Make sure the dimensions weren't changed:
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