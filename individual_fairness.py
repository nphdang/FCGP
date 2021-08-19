import numpy as np

# y_true: true testing labels y_test
# y_pred: predicted testing labels y_pred_testing_round
def generalized_entropy_index(y_true, y_pred, alpha=2):
    """
    Generalized entropy index is proposed as a unified individual and
    group fairness measure in [3]_.  With :math:`b_i = \hat{y}_i - y_i + 1`:
    .. math::
       \mathcal{E}(\alpha) = \begin{cases}
           \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
           \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
           -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
       \end{cases}

    Args:
        alpha (int): Parameter that regulates the weight given to distances
            between values at different parts of the distribution.

    References:
        .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
           "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
           ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
    """

    y_true = (y_true == 1).astype(np.float64)
    y_pred = (y_pred == 1).astype(np.float64)
    b = 1 + y_pred - y_true

    if alpha == 1:
        # moving the b inside the log allows for 0 values
        return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
    elif alpha == 0:
        return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
    else:
        return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))

