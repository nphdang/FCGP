import numpy as np
import copy
import timeit
import datetime
import argparse
import sklearn.gaussian_process as gp
from scipy.linalg import cholesky, cho_solve, solve_triangular
import warnings
import read_data
import group_fairness
import individual_fairness

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="german", type=str, nargs='?', help='dataset')
parser.add_argument('--sensitive', default="age", type=str, nargs='?', help='sensitive')
parser.add_argument('--validation_size', default="0.05", type=str, nargs='?', help='validation size')
parser.add_argument('--test_size', default="0.05", type=str, nargs='?', help='test size')
parser.add_argument('--balance', default="0.5", type=str, nargs='?', help='trade-off factor between accuracy & fairness')
parser.add_argument('--optimize', default="grid", type=str, nargs='?', help='optimization method')
parser.add_argument('--budget', default="50", type=str, nargs='?', help='budget')
parser.add_argument('--n_run', default="5", type=str, nargs='?', help='no of running times')
args = parser.parse_args()
print("dataset: {}, sensitive: {}, validation_size: {}, test_size: {}, "
      "balance: {}, optimize: {}, budget: {}, n_run: {}".
      format(args.dataset, args.sensitive, args.validation_size, args.test_size,
             args.balance, args.optimize, args.budget, args.n_run))

# fit a GP to all samples in validation set,
# compute difference (L1 norm) between mean function (i.e. relabeling function) and initial function on validation set
# compute fairness of relabeling function on validation set
# optimize relabeling function <=> optimize mean function <=> optimize noise variance of using grid search/BO
# we optimize relabeling function to balance two objectives: (1) relabeling function is close to initial function,
# (2) relabeling function improves fairness
# to avoid fitting a new GP at each iteration, we compute mean function using the formula of predictive distribution,
# which depends directly on noise variance
# after obtaining optimal noise variance, we build a GP with optimal noise variance on testing set
# and apply it to predict labels of samples in testing set

dataset = args.dataset
sensitive = args.sensitive
valid_size = float(args.validation_size)
test_size = float(args.test_size)
balance = float(args.balance) # trade-off factor
optimize = args.optimize # grid, bo
budget = int(args.budget) # no of iterations
n_run = int(args.n_run)
method = "fcgp_s"
load_folder_model = "initial_model"
save_folder_model = "relabel_model"

# hyper-parameters to optimize noise_gp
noise_gp_lower = 1e-5
noise_gp_upper = 0.5
n_dim = 1

# compute mean function directly based on noise variance
def predict_w_noise(model_gp_, X_test_, noise_gp_):
    # model_gp_ is GP built on X_test_
    # X_test_ is validation/testing set
    # noise_gp_ noise used in GP

    # 1. compute mean function
    K = model_gp_.kernel_(model_gp_.X_train_)
    K[np.diag_indices_from(K)] += noise_gp_
    try:
        L_ = cholesky(K, lower=True)
        # self.L_ changed, self._K_inv needs to be recomputed
        _K_inv = None
    except np.linalg.LinAlgError as exc:
        exc.args = ("The kernel, %s, is not returning a "
                    "positive definite matrix. Try gradually "
                    "increasing the 'noise' parameter of your "
                    "GaussianProcessRegressor estimator."
                    % model_gp_.kernel_,) + exc.args
        raise
    dual_coef = cho_solve((L_, True), model_gp_.y_train_)
    # compute mean function
    K_trans = model_gp_.kernel_(X_test_, model_gp_.X_train_)
    y_mean = K_trans.dot(dual_coef)
    # undo normalization
    y_mean = model_gp_._y_train_std * y_mean + model_gp_._y_train_mean

    # 2. compute variance function
    # cache result of K_inv computation
    if _K_inv is None:
        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(L_.T, np.eye(L_.shape[0]))
        _K_inv = L_inv.dot(L_inv.T)
    # compute variance of predictive distribution
    y_var = model_gp_.kernel_.diag(X_test_)
    y_var -= np.einsum("ij,ij->i", np.dot(K_trans, _K_inv), K_trans)
    # check if any of the variances is negative because of numerical issues. if yes: set the variance to 0.
    y_var_negative = y_var < 0
    if np.any(y_var_negative):
        warnings.warn("Predicted variances smaller than 0. "
                      "Setting those variances to 0.")
        y_var[y_var_negative] = 0.0
    # undo normalization
    y_var = y_var * model_gp_._y_train_std ** 2

    return y_mean, np.sqrt(y_var)

# objective function to optimize noise variance in GP (i.e. we consider predicted score of initial model is noisy)
def objective_function(noises_to_optimize):
    global acc_bo, fair_bo, theil_bo
    global relabel_bo
    global cnt_obj_func_optimize
    noises_to_optimize = np.array(noises_to_optimize).reshape(-1, n_dim)  # format noises_to_optimize to [[]]
    res = []
    for noise_to_optimize in noises_to_optimize:
        if optimize == "bo": # bo uses log scale
            noise_gp = np.exp(noise_to_optimize[0])
        else: # grid uses original scale
            noise_gp = noise_to_optimize[0]
        # fix error in predict_w_noise() when noise_gp is negative
        if noise_gp < 0:
            noise_gp = noise_gp_lower
        print("noise_gp in objective function: {}".format(round(noise_gp, 4)))
        # compute the difference between relabeling function (i.e. mean function of GP) and initial function on validation set
        ini_pred_validation = copy.deepcopy(y_pred_validation)
        gen_pred_validation, gen_std_validation = predict_w_noise(model_gp_validation, X_valid, noise_gp)
        gen_pred_validation_round = np.around(gen_pred_validation)
        # reformat gen_pred_validation_round as same as y_valid
        gen_pred_validation_round = np.array(gen_pred_validation_round).reshape(-1, 1)
        difference_valid = np.mean(np.abs(gen_pred_validation - ini_pred_validation))
        # compute accuracy and fairness of relabeling function w.r.t the sensitive feature on validation set
        accuracy_overall_valid, demographic_parity_valid, \
        prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
            = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, gen_pred_validation_round)
        print("relabeling func on validation")
        print("difference={}, accuracy={}, fairness={}".
              format(round(difference_valid, 2), round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2)))
        theil_index_valid = individual_fairness.generalized_entropy_index(y_valid, gen_pred_validation_round)
        print("theil_index={}".format(round(theil_index_valid, 2)))
        acc_bo.append(accuracy_overall_valid)
        fair_bo.append(demographic_parity_valid)
        theil_bo.append(theil_index_valid)
        # save new predicted labels on validation set of relabeling function
        relabel_bo.append(gen_pred_validation_round.reshape(1, -1)[0]) # need to reshape to array to add to array
        # compute score for BO (i.e. maximize both fairness and (1 - difference) between two functions)
        similarity_valid = (1 - difference_valid)
        score_bo = (1 - balance) * demographic_parity_valid + balance * similarity_valid
        print("similarity: {}, fairness: {}, score_bo: {}".
              format(round(similarity_valid, 2), round(demographic_parity_valid, 2), round(score_bo, 2)))
        res.append([score_bo])
        cnt_obj_func_optimize = cnt_obj_func_optimize + 1
    res = np.array(res)

    return res

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# read data
_, _, _, _, sen_var_indices = read_data.from_file(dataset, sensitive)
# get sensitive feature index
sen_idx = sen_var_indices[0]

# accuracy, group_fairness, individual_fairness
acc_valid_fcgp, fair_valid_fcgp, individual_valid_fcgp = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
acc_test_fcgp, fair_test_fcgp, individual_test_fcgp = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
for run in range(n_run):
    print("run={}".format(run))
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

    n_valid = len(y_valid)
    n_test = len(y_test)
    print("n_valid: {}, n_test: {}".format(n_valid, n_test))

    # build GP on samples in validation set only 1 time
    time1 = timeit.default_timer()
    kernel = 1.0 * gp.kernels.RBF(length_scale=1.0)
    model_gp_validation = gp.GaussianProcessRegressor(kernel=kernel,
                                                      alpha=noise_gp_lower,
                                                      n_restarts_optimizer=5,
                                                      # y is in [0, 1] and normalize_y=False makes same scale to plot
                                                      normalize_y=False)
    # fit GP to validation set (x is in validation set but y is prediction of initial function on validation set)
    xp_func = np.array(X_valid)
    yp_func = np.array([y[0] for y in y_pred_validation])
    model_gp_validation.fit(xp_func, yp_func)
    time2 = timeit.default_timer()
    print("runtime of fitting a GP to observations: {}(s)".format(round(time2 - time1, 2)))

    # global variables storing accuracy and fairness at each time we optimize objective function
    acc_bo, fair_bo, theil_bo = [], [], []
    # global variable storing new predicted labels of relabeling function at each time we optimize objective function
    relabel_bo = []
    # global variable counting how many times we optimize objective function
    cnt_obj_func_optimize = 0

    if optimize == "grid":
        # no of noise_gp to search
        num_noise_gp = budget

        # search optimal noise_gp on validation set such that it maximizes score
        # (i.e. it maximizes both accuracy and fairness)
        score_arr = np.zeros(num_noise_gp)
        noise_gp_arr = np.zeros_like(score_arr)
        cnt = 0
        # iterate through possible noise_gp
        for noise in np.linspace(noise_gp_lower, noise_gp_upper, num_noise_gp):
            print("cnt: {}".format(cnt))
            print("current noise_gp: {}".format(round(noise, 4)))
            # use current noise_gp to build GP and relabel samples in validation set
            score_arr[cnt] = objective_function(noise)[0]
            noise_gp_arr[cnt] = noise
            cnt += 1
        bestscore = np.max(score_arr)
        bestx_idx = np.argmax(score_arr)
        bestnoise = noise_gp_arr[bestx_idx]
        bestaccuracy = acc_bo[bestx_idx]
        bestfairness = fair_bo[bestx_idx]
        besttheil = theil_bo[bestx_idx]
        print("best_noise: {}".format(round(bestnoise, 4)))
        print("best_score: {}".format(round(bestscore, 2)))
        print("best_accuracy: {}".format(round(bestaccuracy, 2)))
        print("best_fairness: {}".format(round(bestfairness, 2)))
        print("best_theil: {}".format(round(besttheil, 2)))
        gen_pred_validation_round = relabel_bo[bestx_idx]
        # reformat gen_pred_validation_round as same as y_valid
        gen_pred_validation_round = np.array(gen_pred_validation_round).reshape(-1, 1)

    # get best score for each run on validation set
    acc_valid_fcgp[run] = bestaccuracy
    fair_valid_fcgp[run] = bestfairness
    individual_valid_fcgp[run] = besttheil
    # save new predicted labels of relabeling function to file
    with open("./{}/y_relabel_validation_{}_{}_{}_vs{}_ts{}_{}_{}_run{}.file".
                      format(save_folder_model, method, dataset, sensitive, valid_size, test_size, balance, optimize, run), "wb") as f:
        np.save(f, gen_pred_validation_round)

    # use optimal noise variance with the GP built on validation set to relabel samples in testing set
    gen_pred_testing, _ = predict_w_noise(model_gp_validation, X_test, bestnoise)
    gen_pred_testing_round = np.around(gen_pred_testing)
    # reformat gen_pred_testing_round as same as y_test
    gen_pred_testing_round = np.array(gen_pred_testing_round).reshape(-1, 1)
    # compute accuracy and fairness of relabeling function w.r.t the sensitive feature on testing set
    accuracy_overall_test, demographic_parity_test, \
    prob_favored_pred_positive_test, prob_unfavored_pred_positive_test \
        = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, gen_pred_testing_round)
    print("relabeling func on testing")
    print("accuracy={}, fairness={}".format(round(accuracy_overall_test, 2), round(demographic_parity_test, 2)))
    theil_index_test = individual_fairness.generalized_entropy_index(y_test, gen_pred_testing_round)
    print("theil_index={}".format(round(theil_index_test, 2)))
    acc_test_fcgp[run] = accuracy_overall_test
    fair_test_fcgp[run] = demographic_parity_test
    individual_test_fcgp[run] = theil_index_test
    # save new predicted labels of relabeling function to file
    with open("./{}/y_relabel_testing_{}_{}_{}_vs{}_ts{}_{}_{}_run{}.file".
                      format(save_folder_model, method, dataset, sensitive, valid_size, test_size, balance, optimize, run), "wb") as f:
        np.save(f, gen_pred_testing_round)
# end run

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))

# save result to file
acc_valid, acc_valid_std = round(np.mean(acc_valid_fcgp), 2), round(np.std(acc_valid_fcgp), 2)
fair_valid, fair_valid_std = round(np.mean(fair_valid_fcgp), 2), round(np.std(fair_valid_fcgp), 2)
individual_valid, individual_valid_std = round(np.mean(individual_valid_fcgp), 2), round(np.std(individual_valid_fcgp), 2)
acc_test, acc_test_std = round(np.mean(acc_test_fcgp), 2), round(np.std(acc_test_fcgp), 2)
fair_test, fair_test_std = round(np.mean(fair_test_fcgp), 2), round(np.std(fair_test_fcgp), 2)
individual_test, individual_test_std = round(np.mean(individual_test_fcgp), 2), round(np.std(individual_test_fcgp), 2)

with open('./{}/_{}_{}_{}_vs{}_ts{}_balance_{}_optimize_{}_budget_{}.txt'.
                  format(save_folder_model, method, dataset, sensitive, valid_size, test_size, balance, optimize, budget), 'w') as f:
  f.write("dataset: {}, sensitive: {}, validation_size: {}, test_size: {}\n".format(dataset, sensitive, valid_size, test_size))
  f.write("method: {}, balance: {}, optimize: {}, budget: {}\n".format(method, balance, optimize, budget))
  f.write("acc_valid: {} ({}), fair_valid: {} ({}), individual_valid: {} ({})\n".
          format(acc_valid, acc_valid_std, fair_valid, fair_valid_std, individual_valid, individual_valid_std))
  f.write("acc_test: {} ({}), fair_test: {} ({}), individual_test: {} ({})\n".
          format(acc_test, acc_test_std, fair_test, fair_test_std, individual_test, individual_test_std))
  f.write("start date time: {} and end date time: {}\n".format(start_date_time, end_date_time))
  f.write("runtime: {}(s)\n".format(round(end_time-start_time, 2)))

