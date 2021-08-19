import numpy as np
from sklearn.metrics import accuracy_score

# compute accuracy and fairness (i.e. demographic parity) on testing set
# X_true: testing set X_test
# y_true: true testing labels y_test
# y_pred: predicted testing labels y_pred_testing_round
def compute_accuracy_fairness(X_true, sen_idx, y_true, y_pred):
    # compute overall accuracy
    accuracy_overall = accuracy_score(y_true, y_pred)
    # compute no of samples in favored group
    favored_indices = np.where(X_true[:, sen_idx] == 1)[0]
    n_favored = len(favored_indices)
    # compute no of samples in unfavored group
    unfavored_indices = np.where(X_true[:, sen_idx] == 0)[0]
    n_unfavored = len(unfavored_indices)
    # compute no of favored samples predicted as positive outcome
    n_favored_pred_positive = len(np.where(y_pred[favored_indices] == 1)[0])
    # compute no of unfavored samples predicted as positive outcome
    n_unfavored_pred_positive = len(np.where(y_pred[unfavored_indices] == 1)[0])
    # compute probability of each group predicted as positive outcome
    prob_favored_pred_positive = n_favored_pred_positive / n_favored
    prob_unfavored_pred_positive = n_unfavored_pred_positive / n_unfavored
    # compute fairness (i.e. demographic parity)
    demographic_parity = 1 - np.abs(prob_favored_pred_positive - prob_unfavored_pred_positive)

    return accuracy_overall, demographic_parity, prob_favored_pred_positive, prob_unfavored_pred_positive

