import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context("notebook", font_scale=1.8)
plt.style.use('fivethirtyeight')
import argparse
import read_data
import group_fairness

parser = argparse.ArgumentParser()
parser.add_argument('--set', default="test", type=str, nargs='?', help='set')
parser.add_argument('--validation_size', default="0.05", type=str, nargs='?', help='validation size')
parser.add_argument('--test_size', default="0.05", type=str, nargs='?', help='test size')
parser.add_argument('--balance', default="0.5", type=str, nargs='?', help='trade-off factor between accuracy & fairness')
parser.add_argument('--optimize', default="grid", type=str, nargs='?', help='optimization method')
parser.add_argument('--fair_constraint', default="max", type=str, nargs='?', help='fairness constraint')
parser.add_argument('--n_run', default="5", type=str, nargs='?', help='no of running times')
args = parser.parse_args()
print("set: {}, valid_size: {}, test_size: {}, "
      "balance: {}, optimize: {}, fair_constraint: {}, n_run: {}".
      format(args.set, args.validation_size, args.test_size,
             args.balance, args.optimize, args.fair_constraint, args.n_run))

set = args.set # validation, testing
valid_size = float(args.validation_size)
test_size = float(args.test_size)
balance = float(args.balance) # trade-off factor
optimize = args.optimize # grid, bo
fair_constraint = args.fair_constraint # max or 0.95
n_run = int(args.n_run)
load_folder_model = "initial_model"
save_folder_model = "relabel_model"
save_folder_result = "results"
if set == "valid":
    initial_suffix = "valid"
    relabel_suffix = "validation"
if set == "test":
    initial_suffix = "test"
    relabel_suffix = "testing"

datasets_sensitives = ["german+age", "german+sex",
                       "compas+race", "compas+sex",
                       "bank+age", "bank+marital",
                       "adult+sex", "adult+race"]
datasets = ["german", "german",
            "compas", "compas",
            "bank", "bank",
            "adult", "adult"]
sensitives = ["age", "sex",
              "race", "sex",
              "age", "marital",
              "sex", "race"]

methods = ["Pre-trained", "Random", "ROC", "IGD", "FCGP-S", "FCGP-L"]

accuracy_dataset_method_run, fairness_dataset_method_run = [], []
for ds_id in range(len(datasets_sensitives)):
    # get dataset name and sensitive feature
    dataset = datasets[ds_id]
    sensitive = sensitives[ds_id]
    # read data
    _, _, _, _, sen_var_indices = read_data.from_file(dataset, sensitive)
    # get sensitive feature index
    sen_idx = sen_var_indices[0]

    accuracy_method_run, fairness_method_run = [], []
    for method in methods:
        print("method: {}".format(method))
        accuracy_run, fairness_run = np.zeros(n_run), np.zeros(n_run)
        for run in range(n_run):
            print("run: {}".format(run))
            # load testing set from file
            with open("./{}/X_{}_{}_{}_vs{}_ts{}_run{}.file".
                              format(load_folder_model, initial_suffix, dataset, sensitive,
                                     valid_size, test_size, run), "rb") as f:
                X_test = np.load(f)
            with open("./{}/y_{}_{}_{}_vs{}_ts{}_run{}.file".
                              format(load_folder_model, initial_suffix, dataset, sensitive,
                                     valid_size, test_size, run), "rb") as f:
                y_test = np.load(f)
            if method == "Pre-trained":
                # load initial predicted scores from file
                with open("./{}/y_pred_{}_{}_{}_vs{}_ts{}_run{}.file".
                                  format(load_folder_model, relabel_suffix, dataset, sensitive,
                                         valid_size, test_size, run), "rb") as f:
                    y_pred_testing = np.load(f)
                # compute initial predicted labels on testing set
                y_pred_testing_round = np.around(y_pred_testing)
                accuracy_overall, demographic_parity, _, _ \
                    = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
                accuracy_run[run] = accuracy_overall
                fairness_run[run] = demographic_parity
            elif method == "Random":
                method_name = "random"
                # load new predicted labels of relabeling function from file
                with open("./{}/y_relabel_{}_{}_{}_{}_vs{}_ts{}_run{}.file".
                                  format(save_folder_model, relabel_suffix, method_name, dataset, sensitive,
                                         valid_size, test_size, run), "rb") as f:
                    y_pred_testing_round = np.load(f)
                accuracy_overall, demographic_parity, _, _ \
                    = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
                accuracy_run[run] = accuracy_overall
                fairness_run[run] = demographic_parity
            elif method == "ROC":
                method_name = "roc"
                # load new predicted labels of relabeling function from file
                with open("./{}/y_relabel_{}_{}_{}_{}_vs{}_ts{}_fair_{}_run{}.file".
                                  format(save_folder_model, relabel_suffix, method_name, dataset, sensitive,
                                         valid_size, test_size, fair_constraint, run), "rb") as f:
                    y_pred_testing_round = np.load(f)
                accuracy_overall, demographic_parity, _, _ \
                    = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
                accuracy_run[run] = accuracy_overall
                fairness_run[run] = demographic_parity
            elif method == "IGD":
                method_name = "igd"
                # load new predicted labels of relabeling function from file
                with open("./{}/y_relabel_{}_{}_{}_{}_vs{}_ts{}_fair_{}_run{}.file".
                                  format(save_folder_model, relabel_suffix, method_name, dataset, sensitive,
                                         valid_size, test_size, fair_constraint, run), "rb") as f:
                    y_pred_testing_round = np.load(f)
                accuracy_overall, demographic_parity, _, _ \
                    = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
                accuracy_run[run] = accuracy_overall
                fairness_run[run] = demographic_parity
            elif method == "FCGP-S":
                method_name = "fcgp_s"
                # load new predicted labels of relabeling function from file
                with open("./{}/y_relabel_{}_{}_{}_{}_vs{}_ts{}_{}_{}_run{}.file".
                                  format(save_folder_model, relabel_suffix, method_name, dataset, sensitive,
                                         valid_size, test_size, balance, optimize, run), "rb") as f:
                    y_pred_testing_round = np.load(f)
                accuracy_overall, demographic_parity, _, _ \
                    = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
                accuracy_run[run] = accuracy_overall
                fairness_run[run] = demographic_parity
            elif method == "FCGP-L":
                method_name = "fcgp_l"
                # load new predicted labels of relabeling function from file
                with open("./{}/y_relabel_{}_{}_{}_{}_vs{}_ts{}_{}_{}_run{}.file".
                                  format(save_folder_model, relabel_suffix, method_name, dataset, sensitive,
                                         valid_size, test_size, balance, optimize, run), "rb") as f:
                    y_pred_testing_round = np.load(f)
                accuracy_overall, demographic_parity, _, _ \
                    = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
                accuracy_run[run] = accuracy_overall
                fairness_run[run] = demographic_parity
        # store accuracy and fairness of n_run of each method
        accuracy_method_run.append(accuracy_run)
        fairness_method_run.append(fairness_run)
    # store accuracy and fairness of n_run of all methods of each dataset
    accuracy_dataset_method_run.append(accuracy_method_run)
    fairness_dataset_method_run.append(fairness_method_run)

n_dataset = len(datasets_sensitives)
n_method = len(methods)
# compute standard deviation of each method on each dataset
all_accuracy_std, all_fairness_std = [], []
for method_id in range(n_method):
    for data_id in range(n_dataset):
        acc_std = round(np.std(accuracy_dataset_method_run[data_id][method_id]), 2)
        fair_std = round(np.std(fairness_dataset_method_run[data_id][method_id]), 2)
        all_accuracy_std.append(acc_std)
        all_fairness_std.append(fair_std)

# save all results to csv file
file_result = './{}/_result_{}_vs{}_ts{}_{}_{}_{}_nrun{}.csv'.\
    format(save_folder_result, set, valid_size, test_size, balance, optimize, fair_constraint, n_run)
with open(file_result, 'w') as f:
    f.write("dataset,method,run,accuracy,fairness\n")
    for data_id in range(n_dataset):
        for method_id in range(n_method):
            for run_id in range(n_run):
                data_name = datasets_sensitives[data_id]
                method_name = methods[method_id]
                line = data_name + "," + method_name + "," + str(run_id) + "," + \
                       str(accuracy_dataset_method_run[data_id][method_id][run_id]) + "," + \
                       str(fairness_dataset_method_run[data_id][method_id][run_id]) + "\n"
                f.write(line)

# plot results of all methods
df = pd.read_csv(file_result, header=0, sep=",")
print("plot accuracy")
g = sns.catplot(x="dataset", y="accuracy", hue="method", data=df, kind="bar", ci="sd", height=5, aspect=5, palette="Set1")
g.set_xlabels("")
g.set_ylabels("accuracy")
for idx, p in enumerate(g.ax.patches):
    height = round(p.get_height(), 2)
    std = all_accuracy_std[idx]
    print("val: {}, std: {}".format(height, std))
    g.ax.text(p.get_x()+p.get_width()/2, height+std+0.01, str(round(height, 2)), ha="center", fontsize=8)

if set == "valid":
    plt.title("Accuracy on Training Set")
if set == "test":
    plt.title("Accuracy on Test Set")
plt.savefig("./{}/_plot_{}_accuracy_vs{}_ts{}_{}_{}_{}_nrun{}.pdf".
            format(save_folder_result, set, valid_size, test_size,
                   balance, optimize, fair_constraint, n_run), bbox_inches="tight")
plt.close()

print("plot fairness")
g = sns.catplot(x="dataset", y="fairness", hue="method", data=df, kind="bar", ci="sd", height=5, aspect=5, palette="Set1")
g.set_xlabels("")
g.set_ylabels("fairness")
for idx, p in enumerate(g.ax.patches):
    height = round(p.get_height(), 2)
    std = all_fairness_std[idx]
    print("val: {}, std: {}".format(height, std))
    g.ax.text(p.get_x()+p.get_width()/2, height+std+0.01, str(round(height, 2)), ha="center", fontsize=8)

if set == "valid":
    plt.title("Fairness on Training Set")
if set == "test":
    plt.title("Fairness on Test Set")
plt.savefig("./{}/_plot_{}_fairness_vs{}_ts{}_{}_{}_{}_nrun{}.pdf".
            format(save_folder_result, set, valid_size, test_size,
                   balance, optimize, fair_constraint, n_run), bbox_inches="tight")
plt.close()

