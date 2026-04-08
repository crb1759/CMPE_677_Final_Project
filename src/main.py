# ----------------------------------------------------------------
# File: main.py
# Function: Individual Model Setup File
# Author: Connor Bastian
# Date: April 6, 2026
# ----------------------------------------------------------------

# Core Libraries
import pandas as pd
import numpy as np
import math
import time
import itertools

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Scikit-learn
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#Local Imports
import src_parameters as sp

# Model Selection Mapping
MODEL_MAP = {
    "randomforest": ["rf", "randomforest", "random forest"],
    "decisiontree": ["dt", "decisiontree", "decision tree"],
    "linearregression": ["lr", "linearregression", "linear regression"]
}


# -----------------------------
#    Common Model Functions
# -----------------------------

def read_file():
    """
    Function     :
    Parameter(s) :
    Return(s)    :
    """

    df = pd.read_csv("Video Games Sales (1980-2024) - Raw.csv")

    df.head()
    dataframe_shape = df.shape

    print("Read Raw Data with Shape of : ", dataframe_shape)

    return df


def trim_user_input(user_input):
    """
    Function     : Trims the Input from the User to Be Further Processed
    Parameter(s) :
        user_input - User Typed Message
    Return(s)    : A Stripped Version of user_input
    """

    return user_input.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def null_matrices(data):
    """
    Function     :
    Parameter(s) :
    Return(s)    :
    """

    missing = data.isnull().sum()
    missing_statement = pd.DataFrame({
        'Unique Value Count': data.nunique(),
        'Null Value Count': missing,
    })

    return missing_statement


def preprocessing_data(df):
    """
    Function     :
    Parameter(s) :
    Return(s)    :
    """

    print("Beginning Data Preprocessing...")

    df['release_date'] = pd.to_datetime(df['release_date'], dayfirst=True, errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df.head()

    df = df.drop(
        columns=["title", "img", "other_sales",
                 "na_sales", "jp_sales", "pal_sales",
                 "release_date", "last_update"])
    df.head()

    df = df.dropna(subset=['developer'])
    df = df.dropna(subset=['release_year'])
    df = df.dropna(subset=['total_sales'])
    df = df[df["total_sales"] != 0]
    df = df.reset_index(drop=True)

    developer_genre = df.groupby(['genre', 'developer'])['critic_score'].transform('mean')
    publisher_genre = df.groupby(['genre', 'publisher'])['critic_score'].transform('mean')
    console_genre = df.groupby(['genre', 'console'])['critic_score'].transform('mean')

    combined_df = pd.concat([developer_genre, publisher_genre, console_genre], axis=1)
    combined_score = combined_df.mean(axis=1)

    df['critic_score'] = df['critic_score'].fillna(combined_score)

    if df['critic_score'].isnull().any():
        df['critic_score'] = df['critic_score'].fillna(df.groupby('genre')['critic_score'].transform('mean'))

    df['critic_score'] = df['critic_score'].round(1)

    missing = null_matrices(df)
    dataframe_shape = df.shape

    return df


def execute_distributions(df):
    """
    Function     :
    Parameter(s) :
    Return(s)    :
    """

    print("Calculating Data Distributions...")

    user_input = input("Would You Like to View Distributions Upon the Data? (Y/N) : ")

    distribution_flag = False

    if user_input.lower() in ["y", "yes"]:
        distribution_flag = True

    while distribution_flag:

        user_input = input(
            "Enter Distribution to Display "
            "\n\t(Genre by Platform - gxp) "
            "\n\t(Genre by Score - gxs)"
            "\n\t(Genre by Release Year - gxr)"
            "\n\tor 'q' to Quit: "
        )

        if user_input.lower() == 'q':
            print("Exiting Program...")
            break

        distribution_name = user_input

        if distribution_name.lower() == "gxp":
            ct = pd.crosstab(df['console'], df['genre'])
            mask = (ct == 0)
            plt.figure(figsize=(28, 20))
            sns.heatmap(ct, annot=True, fmt="d", cmap="Spectral", linewidths=.5, vmax=500, mask=mask)
            plt.title('Distribution of Genres by Platforms', fontsize=16)
            plt.xlabel('Genre', fontsize=12)
            plt.ylabel('Platform', fontsize=12)

            plt.show()

        if distribution_name.lower() == "gxs":
            ct = pd.crosstab(df['critic_score'], df['genre'])
            mask = (ct == 0)
            plt.figure(figsize=(28, 20))

            sns.heatmap(ct, annot=True, fmt="d", cmap="Spectral", linewidths=.5, vmax=500, mask=mask)

            plt.title('Distribution of Genres by Scores', fontsize=16)
            plt.xlabel('Genre', fontsize=12)
            plt.ylabel('Score', fontsize=12)

            plt.show()

        if distribution_name.lower() == "gxr":
            ct = pd.crosstab(df['release_year'], df['genre'])
            mask = (ct == 0)
            plt.figure(figsize=(28, 20))

            sns.heatmap(ct, annot=True, fmt="d", cmap="Spectral", linewidths=.5, vmax=500, mask=mask)

            plt.title('Distribution of Genres by Release Years', fontsize=16)
            plt.xlabel('Genre', fontsize=12)
            plt.ylabel('Release Year', fontsize=12)

            plt.show()


# -----------------------------
# Random Forest Functionality
# -----------------------------

def rf_train_test(df):
    """
    Function     :
    Parameter(s) :
    Return(s)    :
    """

    y = df["total_sales"]
    Xt = df.drop(columns=["total_sales"])

    low_cardinality_cols = ['genre', 'console', 'release_year']

    high_cardinality_cols = ['publisher', 'developer', 'critic_score']

    preprocessor = ColumnTransformer(
        transformers=[
            ('low_card', OneHotEncoder(handle_unknown='ignore'), low_cardinality_cols),
            ('high_card', TargetEncoder(), high_cardinality_cols)
        ],
        remainder='passthrough' # If there are numeric values already do not touch them
    )

    X = preprocessor.fit_transform(Xt, y)

    feature_count = Xt.shape[1]

    classes = [0, 0.1, 1, np.inf]
    labels = [3, 2, 1]

    data_between_zero_and_one_tenth = df['total_sales'].between(0.0, 0.09999, inclusive='both').sum()
    data_between_one_tenth_and_five_tenth = df['total_sales'].between(0.1, 0.9999, inclusive='both').sum()
    data_greater_than_one = df['total_sales'].between(1, 9999, inclusive='both').sum()

    y = pd.cut(y, bins=classes, labels=labels, right=True)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return feature_count, x_train, x_test, y_train, y_test


def rf_training_plotting(feature_count, x_train, x_test, y_train, y_test):
    """
    Function     :
    Parameter(s) :
    Return(s)    :
    """

    n_estimators_list = [feature_count,feature_count*2, feature_count*5, feature_count*10]
    depths = [3, 5, 10, 0]
    criterions = ["gini", "entropy"]
    is_bootstrap = [True, False]

    # Constant hyperparameters.
    constant_params = {
        'n_jobs': -1,
        'class_weight':'balanced',
        'random_state': 42,
    }

    # Iteration amount of re-train the current combination.
    n_iterations = sp.RF_NUM_ITERATIONS

    best_acc = -float('inf')
    best_params = None

    param_combinations = list(itertools.product(n_estimators_list, depths, criterions, is_bootstrap))

    print("\n--- TRAINING STARTED ---\n")
    print(f"Total {len(param_combinations)} different parameter combination will be tried.\n")

    for n_est, d, c, ibt in param_combinations:

        acc_list = []

        start_time = time.time()

        for i in range(n_iterations):

            model = RandomForestClassifier(
                n_estimators=n_est,
                criterion = c,
                max_depth=None if d==0 else d,
                bootstrap=ibt,
                **constant_params
            )
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            acc = accuracy_score(y_test, y_pred)

            acc_list.append(acc)

        end_time = time.time()
        elapsed_time = end_time - start_time

        avg_acc = np.mean(acc_list)

        print(f"Estimators: {n_est:<4} | Criterion: {'g' if c=='gini' else 'e'} "
              f"| Depth: {'N' if d==0 else 'X' if d==10 else d} | Bootstrap: {1 if ibt else 0} "
              f"| Avg. Acc: {avg_acc:.4f} | Time: {elapsed_time:.2f} sec")

        eff_acc = ( (avg_acc*10) * math.log(n_est) * math.log(max(d,feature_count)) - math.log(elapsed_time) )

        if eff_acc > best_acc:
            best_acc = eff_acc
            best_params = {
                'n_estimators': n_est,
                'criterion': c,
                'max_depth': "None" if d==0 else d,
                'bootstrap': ibt
            }

    print(f"\n--- PARAMETER RESEARCH COMPLETED ---")

    print(f"\nWinner Parameters: {best_params}")
    print(f"Winner's Efficiency Score:  {best_acc:.4f}")


# -----------------------------
#  Base Model Logic Functions
# -----------------------------
def random_forest_regressor(df, num_estimators):

    preprocessed_df = preprocessing_data(df)

    execute_distributions(preprocessed_df)

    feature_count, x_train, x_test, y_train, y_test = rf_train_test(preprocessed_df)

    rf_training_plotting(feature_count, x_train, x_test, y_train, y_test)

    return "Random Forest Hit :", num_estimators


def decision_tree_regressor(df):
    return "Decision Tree Hit"


def linear_regression(df):
    return "Linear Regression Hit"


def get_model_choice(user_input):
    trimmed_user_input = trim_user_input(user_input)

    for key, variations in MODEL_MAP.items():
        if trimmed_user_input in variations:
            return key

    return None


def main_prompt():
    while True:
        user_input = input(
            "Enter Model (Random Forest, Decision Tree, Linear Regression) or 'q' to Quit: "
        )

        if user_input.lower() == 'q':
            print("Exiting Program...")
            break

        data = read_file()

        model_choice = get_model_choice(user_input)

        if model_choice == "randomforest":

            user_input = input(
                "Enter Number of Estimators to Use or 'q' to Quit: "
            )

            if user_input.lower() == 'q':
                print("Exiting Program...")
                break

            num_estimators = user_input

            print("Using Random Forest...")
            model = random_forest_regressor(data, num_estimators)

        elif model_choice == "decisiontree":
            print("Using Decision Tree...")
            model = decision_tree_regressor(data)

        elif model_choice == "linearregression":
            print("Using Linear Regression...")
            model = linear_regression(data)

        else:
            print("Invalid Model Choice. Defaulting to Decision Tree.")
            model = decision_tree_regressor(data)

        print(model)


if __name__ == "__main__":
    main_prompt()
