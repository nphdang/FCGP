import numpy as np
import pandas as pd

# read data from file
def from_file(dataset, sensitive):
    df = pd.read_csv("./data/{}.csv".format(dataset), header=0, sep=",")
    if dataset == "adult":
        numerical_var_names = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        categorical_var_names = ["workclass", "education", "marital-status", "occupation", "relationship", "native-country"]
        categorical_var_names.append(sensitive)
        if sensitive == "race":
            sensitive_var_names = ["race"]
            sensitive_var_values = ["race_White"]
        if sensitive == "sex":
            sensitive_var_names = ["sex"]
            sensitive_var_values = ["sex_Male"]
        class_var_name = "income-per-year"
        class_favored = ">50K"
    if dataset == "german":
        numerical_var_names = ["month", "credit_amount", "investment_as_income_percentage", "residence_since",
                               "number_of_credits", "people_liable_for"]
        categorical_var_names = ["status", "credit_history", "purpose", "savings", "employment", "other_debtors",
                                 "property", "installment_plans", "housing", "skill_level", "telephone", "foreign_worker"]
        categorical_var_names.append(sensitive)
        if sensitive == "age":
            sensitive_var_names = ["age"]
            sensitive_var_values = ["age_adult"]
        if sensitive == "sex":
            sensitive_var_names = ["sex"]
            sensitive_var_values = ["sex_male"]
        class_var_name = "credit"
        class_favored = "1"  # good credit
    if dataset == "compas":
        numerical_var_names = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
        categorical_var_names = ["age_cat", "c_charge_degree", "c_charge_desc"]
        categorical_var_names.append(sensitive)
        if sensitive == "race":
            sensitive_var_names = ["race"]
            sensitive_var_values = ["race_Caucasian"]
        if sensitive == "sex":
            sensitive_var_names = ["sex"]
            sensitive_var_values = ["sex_Male"]
        class_var_name = "two_year_recid"
        class_favored = "0"  # not rearrested
    if dataset == "bank":
        numerical_var_names = ["balance", "duration", "campaign", "pdays", "previous"]
        categorical_var_names = ["job", "education", "default", "housing", "loan", "contact", "poutcome"]
        categorical_var_names.append(sensitive)
        if sensitive == "age":
            sensitive_var_names = ["age"]
            sensitive_var_values = ["age_adult"]
        if sensitive == "marital":
            sensitive_var_names = ["marital"]
            sensitive_var_values = ["marital_single"]
        class_var_name = "subscribe"
        class_favored = "yes"  # yes

    # create dataset with numeric variables
    X = df.loc[:, numerical_var_names]
    # convert categorical variables to discrete variables
    for categorical_var_name in categorical_var_names:
        categorical_var = pd.Categorical(df.loc[:, categorical_var_name])
        # set one dummy variable if it's boolean and not in sensitive variables
        if (len(categorical_var.categories) == 2) & (categorical_var_name != sensitive_var_names[0]):
            drop_first = True
        else:
            drop_first = False
        dummies = pd.get_dummies(categorical_var, prefix=categorical_var_name, drop_first=drop_first)
        # if the variable is binary and in sensitive variables, use sensitive value as dummy variable
        if (len(categorical_var.categories) == 2) & (categorical_var_name == sensitive_var_names[0]):
            dummies = dummies[sensitive_var_values[0]]
        X = pd.concat([X, dummies], axis=1)
    # get label: 1 means "positive" and 0 means "negative"
    labels = pd.Categorical(df.loc[:, class_var_name])
    if labels.categories[1] == class_favored:
        y = np.copy(labels.codes)
    else:
        y = np.copy(labels.codes)
        y[y == 1] = -1
        y[y == 0] = 1
        y[y == -1] = 0
    # get feature names
    feature_names = X.columns.values
    # find indices of sensitive features (column indices in X)
    # in a sensitive feature, 1 means "favored/majority" and 0 means "unfavored/minority"
    sen_var_indices = []
    for sensitive_var_value in sensitive_var_values:
        sen_var_indices.append(np.where(X.columns.values == sensitive_var_value)[0][0])
    sen_var_indices = np.array(sen_var_indices, dtype=int)
    # convert X and y to array
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    n_data = len(y)
    n_feature = len(feature_names)
    print("dataset={}, n_data={}, n_feature={}, sensitive_feature_index={}, sensitive_feature_name={}".
          format(dataset, n_data, n_feature, sen_var_indices[0], sensitive_var_names[0]))

    return X, y, n_data, n_feature, sen_var_indices


# X, y, n_data, n_feature, sen_var_indices = from_file("german", "sex")
