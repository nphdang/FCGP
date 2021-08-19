import numpy as np
import timeit
import datetime
import copy
import argparse
from keras.models import load_model
import read_data
import group_fairness
import individual_fairness

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="german", type=str, nargs='?', help='dataset')
parser.add_argument('--sensitive', default="age", type=str, nargs='?', help='sensitive feature')
parser.add_argument('--validation_size', default="0.05", type=str, nargs='?', help='validation size')
parser.add_argument('--test_size', default="0.05", type=str, nargs='?', help='test size')
parser.add_argument('--method', default="roc", type=str, nargs='?', help='method')
parser.add_argument('--budget', default="50", type=str, nargs='?', help='budget')
parser.add_argument('--fair_constraint', default="max", type=str, nargs='?', help='fairness constraint')
parser.add_argument('--n_run', default="5", type=str, nargs='?', help='no of running times')
args = parser.parse_args()
print("dataset: {}, sensitive: {}, validation_size: {}, test_size: {}, "
      "method: {}, budget: {}, fair_constraint: {}, n_run: {}".
      format(args.dataset, args.sensitive, args.validation_size, args.test_size,
             args.method, args.budget, args.fair_constraint, args.n_run))

dataset = args.dataset
sensitive = args.sensitive
valid_size = float(args.validation_size)
test_size = float(args.test_size)
method = args.method # random, roc (reject option based classification), igd (individual group debiasing)
budget = int(args.budget) # no of iterations
fair_constraint = args.fair_constraint # max or 0.95
if fair_constraint != "max":
    fair_bound = float(fair_constraint)
n_run = int(args.n_run)
load_folder_model = "initial_model"
save_folder_model = "relabel_model"

# relabel samples in critical region
def relabel(X_true, y_true, current_ROC_margin, current_class_threshold=0.5):
    # X_true contains samples
    # y_true is predicted scores of initial model
    # y_pred_round is new predicted labels after relabeling
    y_pred_round = np.zeros(len(y_true))

    # find positive samples whose initial predicted scores are greater than current classification threshold
    fav_pred_inds = (y_true > current_class_threshold)
    # find negative samples whose initial predicted scores are smaller than current classification threshold
    unfav_pred_inds = ~fav_pred_inds
    # reformat indices
    fav_pred_inds = np.array(fav_pred_inds).reshape(1, -1)[0]
    unfav_pred_inds = np.array(unfav_pred_inds).reshape(1, -1)[0]
    # assign predicted positive labels to positive samples
    y_pred_round[fav_pred_inds] = 1
    # assign predicted negative labels to negative samples
    y_pred_round[unfav_pred_inds] = 0

    # find samples in critical region around classification boundary
    crit_region_inds = np.logical_and(y_true <= current_class_threshold + current_ROC_margin,
                                      y_true >= current_class_threshold - current_ROC_margin)

    # find favored and unfavored samples
    favored_indices = (X_true[:, sen_idx] == 1)
    unfavored_indices = (X_true[:, sen_idx] == 0)
    # reformat indices
    favored_indices = np.array(favored_indices).reshape(-1, 1)
    unfavored_indices = np.array(unfavored_indices).reshape(-1, 1)

    # relabel samples in critical region
    # favored samples are assigned negative labels whereas unfavored samples are assigned positive labels
    crit_favored_indices = np.logical_and(crit_region_inds, favored_indices)
    crit_unfavored_indices = np.logical_and(crit_region_inds, unfavored_indices)
    # reformat indices
    crit_favored_indices = np.array(crit_favored_indices).reshape(1, -1)[0]
    crit_unfavored_indices = np.array(crit_unfavored_indices).reshape(1, -1)[0]
    y_pred_round[crit_favored_indices] = 0
    y_pred_round[crit_unfavored_indices] = 1

    return y_pred_round

# debias samples having individual biases
def debias(X_true, y_true, y_true_inverse, indi_bias_scores, current_indi_bias_threshold):
    # X_true contains samples
    # y_true is predicted scores of initial model
    # y_true_inverse is predicted scores of initial model with inverse sensitive feature
    # indi_bias_scores contains individual bias scores of samples in X_true
    # y_pred is new predicted scores after relabeling
    y_pred = copy.deepcopy(y_true)

    # find biased samples whose individual bias scores are greater than current individual bias threshold
    biased_indices = (indi_bias_scores > current_indi_bias_threshold)
    # find unbiased samples whose individual bias scores are smaller than current individual bias threshold
    unbiased_indices = ~biased_indices
    # reformat indices
    biased_indices = np.array(biased_indices).reshape(1, -1)[0]
    unbiased_indices = np.array(unbiased_indices).reshape(1, -1)[0]
    # create individual bias indicators
    indi_bias_indicators = np.zeros(len(indi_bias_scores))
    # assign 1 to biased samples
    indi_bias_indicators[biased_indices] = 1
    # assign 0 to unbiased samples
    indi_bias_indicators[unbiased_indices] = 0

    # NOTE: igd method only focus on relabeling unfavored samples
    # find unfavored samples
    unfavored_indices = (X_true[:, sen_idx] == 0)

    # find biased unfavored samples
    biased_unfavored_indices = np.logical_and(indi_bias_indicators.astype(bool), unfavored_indices)
    # relabel biased unfavored samples
    y_pred[biased_unfavored_indices] = y_true_inverse[biased_unfavored_indices]

    # y_pred_round is new predicted labels after relabeling
    y_pred_round = np.around(y_pred)

    return y_pred_round

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# read data
_, _, _, _, sen_var_indices = read_data.from_file(dataset, sensitive)
# get sensitive feature index
sen_idx = sen_var_indices[0]

# accuracy, group_fairness, individual_fairness
acc_valid_baseline, fair_valid_baseline, individual_valid_baseline = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
acc_test_baseline, fair_test_baseline, individual_test_baseline = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
for run in range(n_run):
    print("run={}".format(run))
    # load initial trained model from file
    trained_model = load_model("./{}/model_{}_{}_vs{}_ts{}_run{}.h5".
                               format(load_folder_model, dataset, sensitive, valid_size, test_size, run))

    # load validation set and prediction from file
    with open("./{}/X_valid_{}_{}_vs{}_ts{}_run{}.file".
                      format(load_folder_model, dataset, sensitive, valid_size, test_size, run), "rb") as f:
        X_valid = np.load(f)
    with open("./{}/y_valid_{}_{}_vs{}_ts{}_run{}.file".
                      format(load_folder_model, dataset, sensitive, valid_size, test_size, run), "rb") as f:
        y_valid = np.load(f)
    with open("./{}/y_pred_validation_{}_{}_vs{}_ts{}_run{}.file".
                      format(load_folder_model, dataset, sensitive, valid_size, test_size, run), "rb") as f:
        y_pred_validation = np.load(f)
    # compute initial predicted labels on validation set
    y_pred_validation_round = np.around(y_pred_validation)
    # compute accuracy and fairness on validation set
    accuracy_overall_valid, demographic_parity_valid, \
    prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
        = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, y_pred_validation_round)

    # load testing set and prediction from file
    with open("./{}/X_test_{}_{}_vs{}_ts{}_run{}.file".
                      format(load_folder_model, dataset, sensitive, valid_size, test_size, run), "rb") as f:
        X_test = np.load(f)
    with open("./{}/y_test_{}_{}_vs{}_ts{}_run{}.file".
                      format(load_folder_model, dataset, sensitive, valid_size, test_size, run), "rb") as f:
        y_test = np.load(f)
    with open("./{}/y_pred_testing_{}_{}_vs{}_ts{}_run{}.file".
                      format(load_folder_model, dataset, sensitive, valid_size, test_size, run), "rb") as f:
        y_pred_testing = np.load(f)
    # compute initial predicted labels on testing set
    y_pred_testing_round = np.around(y_pred_testing)
    # compute accuracy and fairness on testing set
    accuracy_overall_test, demographic_parity_test, \
    prob_favored_pred_positive_test, prob_unfavored_pred_positive_test \
        = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)

    # baseline to improve fairness
    n_valid = len(y_valid)
    n_test = len(y_test)
    print("n_valid: {}, n_test: {}".format(n_valid, n_test))
    if method == "random":
        # select randomly samples from validation/testing set and relabel them to improve fairness
        # no of random samples equals to no of optimization iterations in other methods
        # NOTE: random method works directly on testing set, it doesn't require validation set
        print("relabel validation set")
        sample_indices = np.random.choice(range(n_valid), budget)
        for sample_cnt, sample_idx in enumerate(sample_indices):
            print("sample_cnt: {}, sample_idx: {}".format(sample_cnt, sample_idx))
            # get a random sample
            random_sample = X_valid[sample_idx]
            # get its sensitive feature
            random_sample_sen = random_sample[sen_idx]
            print("random_sample_sen: {}".format(random_sample_sen))
            # more positive outcome for favored group than positive outcome for unfavored group
            if prob_favored_pred_positive_valid > prob_unfavored_pred_positive_valid:
                # this sample belongs to favored group
                if random_sample_sen == 1:
                    # we assign negative outcome to decrease prob_favored_pred_positive
                    y_pred_validation_round[sample_idx] = 0
                # this sample belongs to unfavored group
                elif random_sample_sen == 0:
                    # we assign positive outcome to increase prob_unfavored_pred_positive
                    y_pred_validation_round[sample_idx] = 1
            # less positive outcome for favored group than positive outcome for unfavored group
            if prob_favored_pred_positive_valid < prob_unfavored_pred_positive_valid:
                # this sample belongs to favored group
                if random_sample_sen == 1:
                    # we assign positive outcome to increase prob_favored_pred_positive
                    y_pred_validation_round[sample_idx] = 1
                # this sample belongs to unfavored group
                elif random_sample_sen == 0:
                    # we assign negative outcome to decrease prob_unfavored_pred_positive
                    y_pred_validation_round[sample_idx] = 0
            # re-compute accuracy and fairness on validation set
            accuracy_overall_valid, demographic_parity_valid, \
            prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
                = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, y_pred_validation_round)
            print("relabeling func on validation")
            print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
                  format(round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2),
                         round(prob_favored_pred_positive_valid, 4), round(prob_unfavored_pred_positive_valid, 4)))
            # re-compute theil_index on validation set
            theil_index_valid = individual_fairness.generalized_entropy_index(y_valid, y_pred_validation_round)
            print("theil_index={}".format(round(theil_index_valid, 2)))

        print("relabel testing set")
        sample_indices = np.random.choice(range(n_test), budget)
        for sample_cnt, sample_idx in enumerate(sample_indices):
            print("sample_cnt: {}, sample_idx: {}".format(sample_cnt, sample_idx))
            # get a random sample
            random_sample = X_test[sample_idx]
            # get its sensitive feature
            random_sample_sen = random_sample[sen_idx]
            print("random_sample_sen: {}".format(random_sample_sen))
            # more positive outcome for favored group than positive outcome for unfavored group
            if prob_favored_pred_positive_test > prob_unfavored_pred_positive_test:
                # this sample belongs to favored group
                if random_sample_sen == 1:
                    # we assign negative outcome to decrease prob_favored_pred_positive
                    y_pred_testing_round[sample_idx] = 0
                # this sample belongs to unfavored group
                elif random_sample_sen == 0:
                    # we assign positive outcome to increase prob_unfavored_pred_positive
                    y_pred_testing_round[sample_idx] = 1
            # less positive outcome for favored group than positive outcome for unfavored group
            if prob_favored_pred_positive_test < prob_unfavored_pred_positive_test:
                # this sample belongs to favored group
                if random_sample_sen == 1:
                    # we assign positive outcome to increase prob_favored_pred_positive
                    y_pred_testing_round[sample_idx] = 1
                # this sample belongs to unfavored group
                elif random_sample_sen == 0:
                    # we assign negative outcome to decrease prob_unfavored_pred_positive
                    y_pred_testing_round[sample_idx] = 0
            # re-compute accuracy and fairness on testing set
            accuracy_overall_test, demographic_parity_test, \
            prob_favored_pred_positive_test, prob_unfavored_pred_positive_test \
                = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
            print("relabeling func on testing")
            print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
                  format(round(accuracy_overall_test, 2), round(demographic_parity_test, 2),
                         round(prob_favored_pred_positive_test, 4), round(prob_unfavored_pred_positive_test, 4)))
            # re-compute theil_index on testing set
            theil_index_test = individual_fairness.generalized_entropy_index(y_test, y_pred_testing_round)
            print("theil_index={}".format(round(theil_index_test, 2)))

    # roc fixes optimal classification threshold to 0.5 (default value) and only finds optimal ROC margin
    if method == "roc":
        # search range of ROC margin
        low_ROC_margin = 0.0
        high_ROC_margin = 0.5
        # no of ROC margins to search
        num_ROC_margin = budget
        if fair_constraint != "max":
            # upper and lower bounds of fairness
            metric_ub = 1.0 - fair_bound
            metric_lb = fair_bound - 1.0
        # optimal ROC margin
        optimal_ROC_margin = None

        # step 1: search optimal ROC margin on validation set such that it is small
        # (i.e. no of samples to relabel is small => accuracy is maintained) while
        # fairness score satisfies fairness constraint (i.e. fairness is improved)
        fairness_arr = np.zeros(num_ROC_margin)
        ROC_margin_arr = np.zeros_like(fairness_arr)
        cnt = 0
        # iterate through possible ROC margins
        for ROC_margin in np.linspace(low_ROC_margin, high_ROC_margin, num_ROC_margin):
            print("cnt: {}".format(cnt))
            print("current ROC_margin: {}".format(round(ROC_margin, 4)))
            # use current ROC margin to relabel samples in critical region
            y_pred_validation_round = relabel(X_valid, y_pred_validation, ROC_margin)
            # re-compute accuracy and fairness on validation set
            accuracy_overall_valid, demographic_parity_valid, \
            prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
                = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, y_pred_validation_round)
            print("relabeling func on validation")
            print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
                  format(round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2),
                         round(prob_favored_pred_positive_valid, 4), round(prob_unfavored_pred_positive_valid, 4)))
            # re-compute theil_index on validation set
            theil_index_valid = individual_fairness.generalized_entropy_index(y_valid, y_pred_validation_round)
            print("theil_index={}".format(round(theil_index_valid, 2)))
            # compute fairness with current ROC margin
            # in ROC method, fairness is defined as P(y=positive|S=unprivileged) - P(y=positive|S=privileged)
            fairness_arr[cnt] = prob_unfavored_pred_positive_valid - prob_favored_pred_positive_valid
            ROC_margin_arr[cnt] = ROC_margin
            cnt += 1
        # find good fairness scores that satisfy fairness constraint
        if fair_constraint == "max":
            rel_inds = (np.abs(fairness_arr) == np.min(np.abs(fairness_arr)))
        else:
            rel_inds = np.logical_and(fairness_arr >= metric_lb, fairness_arr <= metric_ub)
        # if we can find some good fairness scores, then get the best one that has possible highest accuracy
        # (i.e. ROC margin is smallest => critical region is smallest => least samples are relabeled)
        if any(rel_inds):
            print("Find some good fairness scores")
            # get good fairness score with smallest ROC margin
            best_ind = np.where(ROC_margin_arr[rel_inds] == np.min(ROC_margin_arr[rel_inds]))[0][0]
        # cannot find any good fairness score satisfying fairness constraint
        # we get best fairness score (i.e. smallest discrimination)
        else:
            print("Cannot find any good fairness score")
            print("fairness_arr: {}".format(fairness_arr))
            rel_inds = np.ones(len(fairness_arr), dtype=bool)
            print("fairness_arr[rel_inds]: {}".format(fairness_arr[rel_inds]))
            best_ind = np.where(np.abs(fairness_arr[rel_inds]) == np.min(np.abs(fairness_arr[rel_inds])))[0][0]
            print("best_ind: {}, smallest_disc: {}".format(best_ind, fairness_arr[rel_inds][best_ind]))
        # get optimal ROC margin
        optimal_ROC_margin = ROC_margin_arr[rel_inds][best_ind]  # get best index among good fairness scores
        print("optimal ROC_margin: {}".format(round(optimal_ROC_margin, 4)))

        # step 2: use optimal ROC margin to relabel samples in validation set and testing set
        print("relabeling func on validation")
        y_pred_validation_round = relabel(X_valid, y_pred_validation, optimal_ROC_margin)
        # reformat y_pred_validation_round as same as y_valid
        y_pred_validation_round = np.array(y_pred_validation_round).reshape(-1, 1)
        # re-compute accuracy and fairness on validation set
        accuracy_overall_valid, demographic_parity_valid, \
        prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
            = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, y_pred_validation_round)
        print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
              format(round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2),
                     round(prob_favored_pred_positive_valid, 4), round(prob_unfavored_pred_positive_valid, 4)))
        # re-compute theil_index on validation set
        theil_index_valid = individual_fairness.generalized_entropy_index(y_valid, y_pred_validation_round)
        print("theil_index={}".format(round(theil_index_valid, 2)))

        print("relabeling func on testing")
        y_pred_testing_round = relabel(X_test, y_pred_testing, optimal_ROC_margin)
        # reformat y_pred_testing_round as same as y_test
        y_pred_testing_round = np.array(y_pred_testing_round).reshape(-1, 1)
        # re-compute accuracy and fairness on testing set
        accuracy_overall_test, demographic_parity_test, \
        prob_favored_pred_positive_test, prob_unfavored_pred_positive_test = \
            group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
        print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
              format(round(accuracy_overall_test, 2), round(demographic_parity_test, 2),
                     round(prob_favored_pred_positive_test, 4), round(prob_unfavored_pred_positive_test, 4)))
        # re-compute theil_index on testing set
        theil_index_test = individual_fairness.generalized_entropy_index(y_test, y_pred_testing_round)
        print("theil_index={}".format(round(theil_index_test, 2)))

    if method == "igd":
        # search range for individual bias threshold
        low_indi_bias_thresh = 0.0
        high_indi_bias_thresh = 1.0
        # no of individual bias thresholds to search
        num_indi_bias_thresh = budget
        if fair_constraint != "max":
            # upper and lower bounds of fairness
            metric_ub = 1.0 - fair_bound
            metric_lb = fair_bound - 1.0
        # optimal individual bias threshold
        optimal_indi_bias_threshold = None

        # step 1: compute individual bias score for each sample in validation set
        # individual bias score of a sample x is the difference in initial predicted score if the sensitive feature of x
        # is set inversely i.e. indi_bias_score(x) = f(x, S=1) - f(x, S=0)

        # create a new validation set where sensitive value is reversed
        X_valid_inverse = copy.deepcopy(X_valid)
        X_valid_inverse[:, sen_idx] = 1 - X_valid_inverse[:, sen_idx]
        # compute initial predicted scores on inverse validation set
        y_pred_validation_inverse = trained_model.predict(X_valid_inverse)
        indi_bias_scores_valid = y_pred_validation_inverse - y_pred_validation

        # step 2: search optimal individual bias threshold on validation set such that
        # individual bias threshold has highest value (i.e. biased samples really have serious individual biases) while
        # group fairness score satisfies group fairness constraint (i.e. group fairness is improved)
        group_fairness_arr = np.zeros(num_indi_bias_thresh)
        indi_bias_thresh_arr = np.zeros_like(group_fairness_arr)
        cnt = 0
        # iterate through possible individual bias thresholds
        for indi_bias_thresh in np.linspace(low_indi_bias_thresh, high_indi_bias_thresh, num_indi_bias_thresh):
            print("cnt: {}".format(cnt))
            print("current indi_bias_threshold: {}".format(round(indi_bias_thresh, 4)))
            # use current individual bias threshold to select biased samples and relabel them
            y_pred_validation_round = debias(X_valid, y_pred_validation, y_pred_validation_inverse,
                                             indi_bias_scores_valid, indi_bias_thresh)
            # re-compute accuracy and fairness on validation set
            accuracy_overall_valid, demographic_parity_valid, \
            prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
                = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, y_pred_validation_round)
            print("relabeling func on validation")
            print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
                  format(round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2),
                         round(prob_favored_pred_positive_valid, 4), round(prob_unfavored_pred_positive_valid, 4)))
            # re-compute theil_index on validation
            theil_index_valid = individual_fairness.generalized_entropy_index(y_valid, y_pred_validation_round)
            print("theil_index={}".format(round(theil_index_valid, 2)))
            # compute group fairness with current individual bias threshold
            # in IGD method, group fairness is defined as P(y=positive|S=unprivileged) - P(y=positive|S=privileged)
            group_fairness_arr[cnt] = prob_unfavored_pred_positive_valid - prob_favored_pred_positive_valid
            indi_bias_thresh_arr[cnt] = indi_bias_thresh
            cnt += 1
        # find good group fairness scores that satisfy group fairness constraint
        if fair_constraint == "max":
            rel_inds = (np.abs(group_fairness_arr) == np.min(np.abs(group_fairness_arr)))
        else:
            rel_inds = np.logical_and(group_fairness_arr >= metric_lb, group_fairness_arr <= metric_ub)
        # if we can find some good group fairness scores, then get the best one that has highest individual bias score
        # since it means that chosen biased samples really have serious individual biases
        if any(rel_inds):
            print("Find some good group fairness scores")
            # get good group fairness score with highest individual bias threshold
            best_ind = np.where(indi_bias_thresh_arr[rel_inds] == np.max(indi_bias_thresh_arr[rel_inds]))[0][0]
        # cannot find any good group fairness score satisfying group fairness constraint
        # we get best group fairness score (i.e. smallest group discrimination)
        else:
            print("Cannot find any good group fairness score")
            print("fairness_arr: {}".format(group_fairness_arr))
            rel_inds = np.ones(len(group_fairness_arr), dtype=bool)
            print("fairness_arr[rel_inds]: {}".format(group_fairness_arr[rel_inds]))
            best_ind = np.where(np.abs(group_fairness_arr[rel_inds]) == np.min(np.abs(group_fairness_arr[rel_inds])))[0][0]
            print("best_ind: {}, smallest_disc: {}".format(best_ind, group_fairness_arr[rel_inds][best_ind]))
        # get optimal individual bias threshold
        optimal_indi_bias_threshold = indi_bias_thresh_arr[rel_inds][best_ind] # get best index among good fairness scores
        print("optimal indi_bias_threshold: {}".format(round(optimal_indi_bias_threshold, 4)))

        # step 3: use optimal individual bias threshold to relabel samples in validation set and testing set
        y_pred_validation_round = debias(X_valid, y_pred_validation, y_pred_validation_inverse,
                                         indi_bias_scores_valid, optimal_indi_bias_threshold)
        # re-compute accuracy and fairness on validation set
        accuracy_overall_valid, demographic_parity_valid, \
        prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
            = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, y_pred_validation_round)
        print("relabeling func on validation")
        print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
              format(round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2),
                     round(prob_favored_pred_positive_valid, 4), round(prob_unfavored_pred_positive_valid, 4)))
        # re-compute theil_index on validation
        theil_index_valid = individual_fairness.generalized_entropy_index(y_valid, y_pred_validation_round)
        print("theil_index={}".format(round(theil_index_valid, 2)))

        # create a new testing set where sensitive value is reversed
        X_test_inverse = copy.deepcopy(X_test)
        X_test_inverse[:, sen_idx] = 1 - X_test_inverse[:, sen_idx]
        # compute initial predicted scores on inverse testing set
        y_pred_testing_inverse = trained_model.predict(X_test_inverse)
        indi_bias_scores_test = y_pred_testing_inverse - y_pred_testing
        y_pred_testing_round = debias(X_test, y_pred_testing, y_pred_testing_inverse,
                                      indi_bias_scores_test, optimal_indi_bias_threshold)
        # re-compute accuracy and fairness on testing set
        accuracy_overall_test, demographic_parity_test, \
        prob_favored_pred_positive_test, prob_unfavored_pred_positive_test \
            = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
        print("relabeling func on testing")
        print("accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
              format(round(accuracy_overall_test, 2), round(demographic_parity_test, 2),
                     round(prob_favored_pred_positive_test, 4), round(prob_unfavored_pred_positive_test, 4)))
        # re-compute theil_index on testing set
        theil_index_test = individual_fairness.generalized_entropy_index(y_test, y_pred_testing_round)
        print("theil_index={}".format(round(theil_index_test, 2)))

    acc_valid_baseline[run] = accuracy_overall_valid
    fair_valid_baseline[run] = demographic_parity_valid
    individual_valid_baseline[run] = theil_index_valid
    acc_test_baseline[run] = accuracy_overall_test
    fair_test_baseline[run] = demographic_parity_test
    individual_test_baseline[run] = theil_index_test

    # save new predicted labels of relabeling function to file
    if method == "random":
        with open("./{}/y_relabel_validation_{}_{}_{}_vs{}_ts{}_run{}.file".
                          format(save_folder_model, method, dataset, sensitive, valid_size, test_size, run), "wb") as f:
            np.save(f, y_pred_validation_round)
        with open("./{}/y_relabel_testing_{}_{}_{}_vs{}_ts{}_run{}.file".
                          format(save_folder_model, method, dataset, sensitive, valid_size, test_size, run), "wb") as f:
            np.save(f, y_pred_testing_round)
    else:
        with open("./{}/y_relabel_validation_{}_{}_{}_vs{}_ts{}_fair_{}_run{}.file".
                          format(save_folder_model, method, dataset, sensitive, valid_size, test_size, fair_constraint, run), "wb") as f:
            np.save(f, y_pred_validation_round)
        with open("./{}/y_relabel_testing_{}_{}_{}_vs{}_ts{}_fair_{}_run{}.file".
                          format(save_folder_model, method, dataset, sensitive, valid_size, test_size, fair_constraint, run), "wb") as f:
            np.save(f, y_pred_testing_round)
# end run

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))

# save result to file
acc_valid, acc_valid_std = round(np.mean(acc_valid_baseline), 2), round(np.std(acc_valid_baseline), 2)
fair_valid, fair_valid_std = round(np.mean(fair_valid_baseline), 2), round(np.std(fair_valid_baseline), 2)
individual_valid, individual_valid_std = round(np.mean(individual_valid_baseline), 2), round(np.std(individual_valid_baseline), 2)
acc_test, acc_test_std = round(np.mean(acc_test_baseline), 2), round(np.std(acc_test_baseline), 2)
fair_test, fair_test_std = round(np.mean(fair_test_baseline), 2), round(np.std(fair_test_baseline), 2)
individual_test, individual_test_std = round(np.mean(individual_test_baseline), 2), round(np.std(individual_test_baseline), 2)
if method == "random":
    file_name = './{}/_{}_{}_{}_vs{}_ts{}_budget_{}.txt'.format(save_folder_model, method, dataset, sensitive,
                                                                valid_size, test_size, budget)
else:
    file_name = './{}/_{}_{}_{}_vs{}_ts{}_budget_{}_fair_{}.txt'.format(save_folder_model, method, dataset, sensitive,
                                                                        valid_size, test_size, budget, fair_constraint)
with open(file_name, 'w') as f:
  f.write("dataset: {}, sensitive: {}, validation_size: {}, test_size: {}\n".format(dataset, sensitive, valid_size, test_size))
  if method == "random":
      f.write("method: {}, budget: {}\n".format(method, budget))
  else:
      f.write("method: {}, budget: {}, fair_constraint: {}\n".format(method, budget, fair_constraint))
  f.write("acc_valid: {} ({}), fair_valid: {} ({}), individual_valid: {} ({})\n".
          format(acc_valid, acc_valid_std, fair_valid, fair_valid_std, individual_valid, individual_valid_std))
  f.write("acc_test: {} ({}), fair_test: {} ({}), individual_test: {} ({})\n".
          format(acc_test, acc_test_std, fair_test, fair_test_std, individual_test, individual_test_std))
  f.write("start date time: {} and end date time: {}\n".format(start_date_time, end_date_time))
  f.write("runtime: {}(s)\n".format(round(end_time-start_time, 2)))

