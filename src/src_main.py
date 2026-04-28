# ----------------------------------------------------------------
# File: main.py
# Function: Video Game Sales ML Project
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import math
import time
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

import src_parameters as sp


MODEL_MAP = {
    "randomforest": ["rf", "randomforest", "random forest"],
    "decisiontree": ["dt", "decisiontree", "decision tree"],
    "linearregression": ["lr", "linearregression", "linear regression", "logisticregression", "logistic regression"],
    "svm": ["svm", "supportvectormachine", "support vector machine"],
    "compare": ["compare", "comparison", "modelcomparison", "model comparison", "cmp", "all"]
}


def read_file():
    df = pd.read_csv("Video Games Sales (1980-2024) - Raw.csv")
    print("Read Raw Data with Shape of:", df.shape)
    return df


def trim_user_input(user_input):
    return user_input.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def null_matrices(data):
    missing = data.isnull().sum()
    return pd.DataFrame({
        "Unique Value Count": data.nunique(),
        "Null Value Count": missing,
    })


def preprocessing_data(df):
    print("Beginning Data Preprocessing...")

    df["release_date"] = pd.to_datetime(df["release_date"], dayfirst=True, errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month  # <-- ADD THIS

    df = df.drop(
        columns=[
            "img", "other_sales", "na_sales", "jp_sales",
            "pal_sales", "release_date", "last_update"
        ],
        errors="ignore"
    )

    df = df.dropna(subset=["developer", "release_year", "total_sales"])
    df = df[df["total_sales"] != 0]
    df = df.reset_index(drop=True)

    developer_genre = df.groupby(["genre", "developer"])["critic_score"].transform("mean")
    publisher_genre = df.groupby(["genre", "publisher"])["critic_score"].transform("mean")
    console_genre = df.groupby(["genre", "console"])["critic_score"].transform("mean")

    combined_score = pd.concat(
        [developer_genre, publisher_genre, console_genre],
        axis=1
    ).mean(axis=1)

    df["critic_score"] = df["critic_score"].fillna(combined_score)

    if df["critic_score"].isnull().any():
        df["critic_score"] = df["critic_score"].fillna(
            df.groupby("genre")["critic_score"].transform("mean")
        )

    df["critic_score"] = df["critic_score"].fillna(df["critic_score"].mean())
    df["critic_score"] = df["critic_score"].round(1)

    # Fill missing release_month with the most common month in that genre
    df["release_month"] = df["release_month"].fillna(
        df.groupby("genre")["release_month"].transform(lambda x: x.mode()[0] if not x.mode().empty else 11)
    ).fillna(11)  # default to November (most common game release month)
    df["release_month"] = df["release_month"].astype(int)

    print("Processed Data Shape:", df.shape)
    print(null_matrices(df))

    return df


def execute_distributions(df):
    user_input = input("Would You Like to View Distributions Upon the Data? (Y/N): ")

    if user_input.lower() not in ["y", "yes"]:
        return

    while True:
        distribution_name = input(
            "Enter Distribution to Display "
            "\n\t(Genre by Platform - gxp)"
            "\n\t(Genre by Score - gxs)"
            "\n\t(Genre by Release Year - gxr)"
            "\n\tor 'q' to Quit: "
        )

        if distribution_name.lower() == "q":
            break

        if distribution_name.lower() == "gxp":
            ct = pd.crosstab(df["console"], df["genre"])
            title = "Distribution of Genres by Platforms"
            x_label = "Genre"
            y_label = "Platform"

        elif distribution_name.lower() == "gxs":
            ct = pd.crosstab(df["critic_score"], df["genre"])
            title = "Distribution of Genres by Scores"
            x_label = "Genre"
            y_label = "Critic Score"

        elif distribution_name.lower() == "gxr":
            ct = pd.crosstab(df["release_year"], df["genre"])
            title = "Distribution of Genres by Release Years"
            x_label = "Genre"
            y_label = "Release Year"

            #attempt at showing genres over time
            genres = ['Action', 'Shooter', 'Action-Adventure', 'Sports', 'Role-Playing', 'Simulation', 'Racing', 'Music', 'Misc', 'Platform', 'Strategy']
            plt.figure(figsize=(10, 35))
            i = 1
            for genre in genres:
                plt.subplot(len(genres), 1, i)
                act = df[df['genre'] == genre]
                gxr = act['release_year']
                counts = gxr.value_counts()
                plt.bar(counts.index, counts.values, color='orange')
                plt.xlabel('Year')
                plt.ylabel('Count')
                plt.title(genre + ' Frequency by year')
                current_values = plt.gca().get_yticks()
                plt.axis([1975, 2020, 0, 250])
                i += 1
            plt.tight_layout()
            plt.show()

        else:
            print("Invalid distribution choice.")
            continue

        mask = ct == 0
        plt.figure(figsize=(28, 20))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Spectral", linewidths=0.5, vmax=500, mask=mask)
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.show()


def prepare_features_and_labels(df):
    y_raw = df["total_sales"]
    X_raw = df.drop(columns=["total_sales"])

    low_cardinality_cols = ["genre", "console", "release_year", "release_month"]  # <-- month added
    high_cardinality_cols = ["publisher", "developer", "critic_score", "title"]   # <-- title added

    preprocessor = ColumnTransformer(
        transformers=[
            ("low_card", OneHotEncoder(handle_unknown="ignore"), low_cardinality_cols),
            ("high_card", TargetEncoder(), high_cardinality_cols)
        ],
        remainder="passthrough"
    )

    bins = [0, 0.1, 1, np.inf]
    labels = [3, 2, 1]

    y = pd.cut(y_raw, bins=bins, labels=labels, right=True)

    X = preprocessor.fit_transform(X_raw, y)

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=sp.RANDOM_STATE
    )

    return X_raw, x_train, x_test, y_train, y_test, preprocessor


def print_evaluation(model, x_test, y_test):
    y_pred = model.predict(x_test)

    print("\n--- Model Evaluation ---")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def compare_models(x_train, x_test, y_train, y_test):

    print("\n--- MODEL COMPARISON STARTED ---")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            criterion="gini",
            max_depth=None,
            bootstrap=True,
            class_weight="balanced",
            random_state=sp.RANDOM_STATE,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            class_weight="balanced",
            random_state=sp.RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=sp.RANDOM_STATE
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=sp.RANDOM_STATE
    )
    }

    results = []

    for name, model in models.items():
        start_time = time.time()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        elapsed_time = time.time() - start_time

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Training Time (sec)": round(elapsed_time, 2)
        })

    results_df = pd.DataFrame(results)

    print("\n--- Model Comparison Results ---")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["Accuracy"].idxmax()]
    print("\nBest Model Based on Accuracy:")
    print(f"{best_row['Model']} with accuracy {best_row['Accuracy']}")

    return results_df


def rf_training_plotting(feature_count, x_train, x_test, y_train, y_test):

    n_estimators_list = [feature_count, feature_count * 2, feature_count * 5, feature_count * 10]
    depths = [3, 5, 10, 0]
    criterions = ["gini", "entropy"]
    is_bootstrap = [True, False]

    constant_params = {
        "n_jobs": -1,
        "class_weight": "balanced",
        "random_state": 42,
    }

    n_iterations = sp.RF_NUM_ITERATIONS

    best_score = -float("inf")
    best_params = None
    best_model = None

    param_combinations = list(itertools.product(
        n_estimators_list,
        depths,
        criterions,
        is_bootstrap
    ))

    print("\n--- RANDOM FOREST TRAINING STARTED ---")
    print(f"Trying {len(param_combinations)} parameter combinations.\n")

    for n_est, d, c, ibt in param_combinations:
        acc_list = []
        start_time = time.time()

        for _ in range(n_iterations):
            model = RandomForestClassifier(
                n_estimators=n_est,
                criterion=c,
                max_depth=None if d == 0 else d,
                bootstrap=ibt,
                **constant_params
            )

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            acc_list.append(acc)

        elapsed_time = time.time() - start_time
        avg_acc = np.mean(acc_list)

        print(
            f"Estimators: {n_est:<4} | Criterion: {'g' if c == 'gini' else 'e'} "
            f"| Depth: {'N' if d == 0 else 'X' if d == 10 else d} "
            f"| Bootstrap: {1 if ibt else 0} "
            f"| Avg. Acc: {avg_acc:.4f} | Time: {elapsed_time:.2f} sec"
        )

        safe_time = max(elapsed_time, 0.0001)
        effective_score = ((avg_acc * 10) * math.log(n_est) * math.log(max(d, feature_count)) - math.log(safe_time))

        if effective_score > best_score:
            best_score = effective_score
            best_params = {
                "n_estimators": n_est,
                "criterion": c,
                "max_depth": None if d == 0 else d,
                "bootstrap": ibt
            }
            best_model = model

    print("\n--- PARAMETER RESEARCH COMPLETED ---")
    print("Winner Parameters:", best_params)
    print(f"Winner Efficiency Score: {best_score:.4f}")

    print_evaluation(best_model, x_test, y_test)

    return best_model


def train_decision_tree(x_train, x_test, y_train, y_test):

    print("\n--- DECISION TREE TRAINING STARTED ---")

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        class_weight="balanced",
        random_state=sp.RANDOM_STATE
    )

    model.fit(x_train, y_train)
    print_evaluation(model, x_test, y_test)

    return model


def train_logistic_regression(x_train, x_test, y_train, y_test):

    print("\n--- LOGISTIC REGRESSION TRAINING STARTED ---")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=sp.RANDOM_STATE
    )

    model.fit(x_train, y_train)
    print_evaluation(model, x_test, y_test)

    return model


def train_svm(x_train, x_test, y_train, y_test):

    print("\n--- SVM TRAINING STARTED ---")

    model = SVC(
        kernel="rbf",          # try 'linear', 'poly', 'rbf'
        C=1.0,                 # regularization
        gamma="scale",         # kernel coefficient
        class_weight="balanced",  # helps with imbalance
        random_state=sp.RANDOM_STATE
    )

    model.fit(x_train, y_train)
    print_evaluation(model, x_test, y_test)

    return model

def interactive_sales_prediction(model, preprocessor, X_raw):
    choice = input("\nRun interactive sales prediction demo? (Y/N): ")
    if choice.lower() not in ["y", "yes"]:
        return

    while True:
        print("\n--- Interactive Video Game Sales Predictor ---")
        print("(Press Enter to skip any field — blanks will use dataset averages)\n")

        title       = input("Game Title: ").strip()
        genre       = input("Genre: ").strip()
        console     = input("Console: ").strip()
        publisher   = input("Publisher: ").strip()
        developer   = input("Developer: ").strip()
        score_input = input("Critic Score: ").strip()
        year_input  = input("Release Year: ").strip()
        month_input = input("Release Month (1-12): ").strip()

        def resolve_cat(user_val, field):
            return user_val if user_val else X_raw[field].mode()[0]

        def resolve_num(user_val, field):
            if user_val:
                try:
                    return float(user_val)
                except ValueError:
                    print(f"  Invalid number for {field}, using dataset average.")
            return X_raw[field].mean()

        resolved = {
            "title":         resolve_cat(title,       "title"),
            "genre":         resolve_cat(genre,        "genre"),
            "console":       resolve_cat(console,      "console"),
            "publisher":     resolve_cat(publisher,    "publisher"),
            "developer":     resolve_cat(developer,    "developer"),
            "critic_score":  round(resolve_num(score_input, "critic_score"), 1),
            "release_year":  int(resolve_num(year_input,  "release_year")),
            "release_month": int(resolve_num(month_input, "release_month")),
        }

        print("\n  Using values:")
        for k, v in resolved.items():
            print(f"    {k:<15}: {v}")

        try:
            sample = pd.DataFrame([resolved])
            sample_encoded = preprocessor.transform(sample)
            prediction = model.predict(sample_encoded)[0]

            label_map = {
                3: "Low Sales    — under 0.1 million",
                2: "Medium Sales — 0.1 to 1 million",
                1: "High Sales   — over 1 million"
            }

            print("\n  Predicted Sales Category:")
            print(f"  >>> {label_map[int(prediction)]} <<<")

        except Exception as e:
            print(f"\n  Prediction failed: {e}")

        again = input("\nPredict another game? (Y/N): ")
        if again.lower() not in ["y", "yes"]:
            return


def setup_project_data(df):
    preprocessed_df = preprocessing_data(df)
    execute_distributions(preprocessed_df)

    X_raw, x_train, x_test, y_train, y_test, preprocessor = prepare_features_and_labels(preprocessed_df)

    return X_raw, x_train, x_test, y_train, y_test, preprocessor


def random_forest_classifier(df, num_estimators):
    X_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    run_comparison = input("\nRun model comparison before Random Forest search? (Y/N): ")
    if run_comparison.lower() in ["y", "yes"]:
        compare_models(x_train, x_test, y_train, y_test)

    feature_count = X_raw.shape[1]

    best_model = rf_training_plotting(
        feature_count,
        x_train,
        x_test,
        y_train,
        y_test
    )

    interactive_sales_prediction(best_model, preprocessor, X_raw)

    return "Random Forest Complete"


def decision_tree_classifier(df):
    x_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    model = train_decision_tree(x_train, x_test, y_train, y_test)
    interactive_sales_prediction(model, preprocessor, x_raw)

    return "Decision Tree Complete"


def linear_regression_classifier(df):
    x_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    model = train_logistic_regression(x_train, x_test, y_train, y_test)
    interactive_sales_prediction(model, preprocessor, x_raw)

    return "Logistic Regression Complete"


def svm_classifier(df):
    x_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    model = train_svm(x_train, x_test, y_train, y_test)
    interactive_sales_prediction(model, preprocessor, x_raw)

    return "SVM Complete"


def compare_all_models(df):
    _, x_train, x_test, y_train, y_test, _ = setup_project_data(df)

    compare_models(x_train, x_test, y_train, y_test)

    return "Model Comparison Complete"


def get_model_choice(user_input):
    trimmed_user_input = trim_user_input(user_input)

    for key, variations in MODEL_MAP.items():
        trimmed_variations = [trim_user_input(v) for v in variations]
        if trimmed_user_input in trimmed_variations:
            return key

    return None


def main_prompt():
    while True:
        user_input = input(
            "\nEnter Model "
            "(Random Forest, Decision Tree, Logistic Regression, SVM, Compare) "
            "or 'q' to Quit: "
        )

        if user_input.lower() == "q":
            print("Exiting Program...")
            break

        data = read_file()
        model_choice = get_model_choice(user_input)

        if model_choice == "randomforest":
            user_input = input("Enter Number of Estimators to Use or press Enter for automatic search: ")

            if user_input.lower() == "q":
                print("Exiting Program...")
                break

            num_estimators = user_input if user_input else "automatic"

            print("Using Random Forest...")
            model = random_forest_classifier(data, num_estimators)

        elif model_choice == "decisiontree":
            print("Using Decision Tree...")
            model = decision_tree_classifier(data)

        elif model_choice == "linearregression":
            print("Using Logistic Regression Classification...")
            model = linear_regression_classifier(data)

        elif model_choice == "svm":
            print("Using Support Vector Machine...")
            model = svm_classifier(data)

        elif model_choice == "compare":
            print("Comparing All Models...")
            model = compare_all_models(data)

        else:
            print("Invalid Model Choice. Defaulting to Decision Tree.")
            model = decision_tree_classifier(data)

        print(model)


if __name__ == "__main__":
    main_prompt()