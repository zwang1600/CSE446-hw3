from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return np.add(np.outer(x_i, x_j), 1)**d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    
    return np.exp(-gamma * np.subtract.outer(x_i, x_j)**2)


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    n = x.size
    K = kernel_function(x, x, kernel_param)
    
    return np.linalg.solve(K + _lambda * np.eye(n), y)


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    loss = 0
    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size
        x_train = np.concatenate([x[:start], x[end:]])
        y_train = np.concatenate([y[:start], y[end:]])
        x_val = x[start:end]
        y_val = y[start:end]

        # Compute kernel matrix for validation set (the shape should be n_val * n_train)
        K = kernel_function(x_val, x_train, kernel_param)
        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        y_hat = np.dot(K, alpha)
        loss += np.mean((y_hat - y_val)**2)
    
    return loss / num_folds


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    # Calculate gamma
    distances = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            distance = np.linalg.norm(x[i] - x[j]) ** 2
            distances.append(distance)
    gamma = 1 / np.median(distances)

    best_lambda = 10e-5
    min_loss = cross_validation(x, y, rbf_kernel, gamma, best_lambda, num_folds)

    for _lambda in np.linspace(10e-5, 10e-1):
        loss = cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)
        # print(f'lambda: {_lambda}  loss: {loss}')
        if loss < min_loss:
            best_lambda = _lambda
            min_loss = loss

    return (best_lambda, gamma)


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    best_d = 5
    best_lambda = 10e-5
    min_loss = cross_validation(x, y, poly_kernel, best_d, best_lambda, num_folds)
    
    for _lambda in np.linspace(10e-5, 10e-1):
        for d in np.linspace(5, 25):
            loss = cross_validation(x, y, poly_kernel, d, _lambda, num_folds)
            # print(f'loss: {loss}  d: {d}   lambda: {_lambda}')
            if loss < min_loss:
                best_lambda = _lambda
                best_d = d
                min_loss = loss

    return (best_lambda, best_d)


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. Repeat A, B with x_300, y_300

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    
    optimal_lambda_rbf, gamma = rbf_param_search(x_30, y_30, 29)
    print(optimal_lambda_rbf)
    print(gamma)
    alpha_rbf = train(x_30, y_30, rbf_kernel, gamma, optimal_lambda_rbf)
    grid = np.linspace(0, 1, num=100)
    rbf_predictions = rbf_kernel(grid, x_30, gamma) @ alpha_rbf
    plt.plot(grid, rbf_predictions, label='RBF Kernel')
    
    # optimal_lambda_ploy, optimal_d = poly_param_search(x_30, y_30, 29)
    # print(optimal_d)
    # print(optimal_lambda_ploy)
    # alpha_poly = train(x_30, y_30, poly_kernel, optimal_d, optimal_lambda_ploy)
    # grid = np.linspace(0, 1, num=100)
    # poly_predictions = poly_kernel(grid, x_30, optimal_d) @ alpha_poly
    # plt.plot(grid, poly_predictions, label='Polynomial Kernel')

    plt.scatter(x_30, y_30, color='#A9A9A9',label='Data')
    plt.plot(grid, f_true(grid), label='True F')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='best', prop={'size': 8})
    plt.ylim(-6, 6)
    plt.show()
    

if __name__ == "__main__":
    main()
