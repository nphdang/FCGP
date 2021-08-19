import numpy as np
import timeit
import datetime
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import read_data
import group_fairness
import individual_fairness

import os
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default="-1", type=str, nargs='?', help='gpu id')
parser.add_argument('--dataset', default="german", type=str, nargs='?', help='dataset')
parser.add_argument('--sensitive', default="age", type=str, nargs='?', help='sensitive feature')
parser.add_argument('--validation_size', default="0.05", type=str, nargs='?', help='validation size')
parser.add_argument('--test_size', default="0.05", type=str, nargs='?', help='test size')
parser.add_argument('--n_run', default="5", type=str, nargs='?', help='no of running times')
args = parser.parse_args()
print("gpu_id: {}, dataset: {}, sensitive: {}, validation_size: {}, test_size: {}, n_run: {}".
      format(args.gpu, args.dataset, args.sensitive, args.validation_size, args.test_size, args.n_run))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

dataset = args.dataset
sensitive = args.sensitive
valid_size = float(args.validation_size)
test_size = float(args.test_size)
n_run = int(args.n_run)
save_folder_model = "initial_model"

### hyper-parameters of NN
hidden_dim = 32
batch_size = 32
epochs = 20

start_date_time = datetime.datetime.now()
start_time = timeit.default_timer()

# read data
X_data, y_data, n_data, n_feature, sen_var_indices = read_data.from_file(dataset, sensitive)
# get sensitive feature index
sen_idx = sen_var_indices[0]
# normalize X
X_data = MinMaxScaler().fit_transform(X_data)

# accuracy, group_fairness, individual_fairness
acc_valid_initial, fair_valid_initial, individual_valid_initial = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
acc_test_initial, fair_test_initial, individual_test_initial = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
for run in range(n_run):
    print("run={}".format(run))
    # split data to training, validation, and testing sets
    X_train_test, X_valid, y_train_test, y_valid = train_test_split(X_data, y_data, test_size=valid_size,
                                                                    random_state=run, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=test_size,
                                                        random_state=run, shuffle=True)
    print("n_train: {}, n_valid: {}, n_test: {}".format(len(y_train), len(y_valid), len(y_test)))

    # train NN once to obtain initial function
    time1 = timeit.default_timer()
    trained_model = Sequential()
    trained_model.add(Dense(hidden_dim, input_dim=n_feature, activation='relu'))
    trained_model.add(Dense(1, activation='sigmoid'))
    trained_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    trained_model.fit(X_train, y_train, shuffle=True, epochs=epochs, batch_size=batch_size, verbose=0)

    # compute accuracy and fairness on validation set
    y_pred_validation = trained_model.predict(X_valid)
    y_pred_validation_round = np.around(y_pred_validation)
    accuracy_overall_valid, demographic_parity_valid, prob_favored_pred_positive_valid, prob_unfavored_pred_positive_valid \
        = group_fairness.compute_accuracy_fairness(X_valid, sen_idx, y_valid, y_pred_validation_round)
    print("initial func on validation: accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
          format(round(accuracy_overall_valid, 2), round(demographic_parity_valid, 2),
                 round(prob_favored_pred_positive_valid, 4), round(prob_unfavored_pred_positive_valid, 4)))
    theil_index_valid = individual_fairness.generalized_entropy_index(y_valid, y_pred_validation_round)
    print("initial func on validation: theil_index={}".format(round(theil_index_valid, 2)))
    acc_valid_initial[run] = accuracy_overall_valid
    fair_valid_initial[run] = demographic_parity_valid
    individual_valid_initial[run] = theil_index_valid
    # save validation set and prediction to file
    with open("./{}/X_valid_{}_{}_vs{}_ts{}_run{}.file".
                      format(save_folder_model, dataset, sensitive, valid_size, test_size, run), "wb") as f:
        np.save(f, X_valid)
    with open("./{}/y_valid_{}_{}_vs{}_ts{}_run{}.file".
                      format(save_folder_model, dataset, sensitive, valid_size, test_size, run), "wb") as f:
        np.save(f, y_valid)
    with open("./{}/y_pred_validation_{}_{}_vs{}_ts{}_run{}.file".
                      format(save_folder_model, dataset, sensitive, valid_size, test_size, run), "wb") as f:
        np.save(f, y_pred_validation)

    # compute accuracy and fairness on testing set
    y_pred_testing = trained_model.predict(X_test)
    y_pred_testing_round = np.around(y_pred_testing)
    accuracy_overall_test, demographic_parity_test, prob_favored_pred_positive_test, prob_unfavored_pred_positive_test \
        = group_fairness.compute_accuracy_fairness(X_test, sen_idx, y_test, y_pred_testing_round)
    print("initial func on testing: accuracy={}, fairness={}, p_favored_positive={}, p_unfavored_positive={}".
          format(round(accuracy_overall_test, 2), round(demographic_parity_test, 2),
                 round(prob_favored_pred_positive_test, 4), round(prob_unfavored_pred_positive_test, 4)))
    theil_index_test = individual_fairness.generalized_entropy_index(y_test, y_pred_testing_round)
    print("initial func on testing: theil_index={}".format(round(theil_index_test, 2)))
    acc_test_initial[run] = accuracy_overall_test
    fair_test_initial[run] = demographic_parity_test
    individual_test_initial[run] = theil_index_test
    # save testing set and prediction to file
    with open("./{}/X_test_{}_{}_vs{}_ts{}_run{}.file".
                      format(save_folder_model, dataset, sensitive, valid_size, test_size, run), "wb") as f:
        np.save(f, X_test)
    with open("./{}/y_test_{}_{}_vs{}_ts{}_run{}.file".
                      format(save_folder_model, dataset, sensitive, valid_size, test_size, run), "wb") as f:
        np.save(f, y_test)
    with open("./{}/y_pred_testing_{}_{}_vs{}_ts{}_run{}.file".
                      format(save_folder_model, dataset, sensitive, valid_size, test_size, run), "wb") as f:
        np.save(f, y_pred_testing)

    # save NN
    trained_model.save("./{}/model_{}_{}_vs{}_ts{}_run{}.h5".
                       format(save_folder_model, dataset, sensitive, valid_size, test_size, run))
    time2 = timeit.default_timer()
    print("runtime of NN: {}(s)".format(round(time2 - time1, 2)))

    del trained_model
    K.clear_session()
# end run

end_date_time = datetime.datetime.now()
end_time = timeit.default_timer()
print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
print("runtime: {}(s)".format(round(end_time-start_time, 2)))

# save result to file
acc_valid, acc_valid_std = round(np.mean(acc_valid_initial), 2), round(np.std(acc_valid_initial), 2)
fair_valid, fair_valid_std = round(np.mean(fair_valid_initial), 2), round(np.std(fair_valid_initial), 2)
individual_valid, individual_valid_std = round(np.mean(individual_valid_initial), 2), round(np.std(individual_valid_initial), 2)
acc_test, acc_test_std = round(np.mean(acc_test_initial), 2), round(np.std(acc_test_initial), 2)
fair_test, fair_test_std = round(np.mean(fair_test_initial), 2), round(np.std(fair_test_initial), 2)
individual_test, individual_test_std = round(np.mean(individual_test_initial), 2), round(np.std(individual_test_initial), 2)
with open('./{}/_model_{}_{}_vs{}_ts{}.txt'.format(save_folder_model, dataset, sensitive, valid_size, test_size), 'w') as f:
  f.write("method: initial, dataset: {}, sensitive: {}\n".format(dataset, sensitive))
  f.write("n_train: {}, n_valid: {}, n_test: {}\n".format(len(y_train), len(y_valid), len(y_test)))
  f.write("acc_valid: {} ({}), fair_valid: {} ({}), individual_valid: {} ({})\n".
          format(acc_valid, acc_valid_std, fair_valid, fair_valid_std, individual_valid, individual_valid_std))
  f.write("acc_test: {} ({}), fair_test: {} ({}), individual_test: {} ({})\n".
          format(acc_test, acc_test_std, fair_test, fair_test_std, individual_test, individual_test_std))
  f.write("start date time: {} and end date time: {}\n".format(start_date_time, end_date_time))
  f.write("runtime: {}(s)\n".format(round(end_time-start_time, 2)))

