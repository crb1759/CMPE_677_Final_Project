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


# Maps canonical model keys to all accepted user input aliases.
# Used by get_model_choice() to resolve free-text input to a model key.
MODEL_MAP = {
    "randomforest": ["rf", "randomforest", "random forest"],
    "decisiontree": ["dt", "decisiontree", "decision tree"],
    "linearregression": ["lr", "linearregression", "linear regression", "logisticregression", "logistic regression"],
    "svm": ["svm", "supportvectormachine", "support vector machine"],
    "compare": ["compare", "comparison", "modelcomparison", "model comparison", "cmp", "c", "all"]
}


def read_file():
    """
    Read the raw video game sales CSV into a DataFrame.

    Returns:
        pd.DataFrame: Raw dataset loaded from disk.
    """
    df = pd.read_csv("Video Games Sales (1980-2024) - Raw.csv")
    print("Read Raw Data with Shape of:", df.shape)
    return df


def trim_user_input(user_input):
    """
    Normalize a user-provided string for case-insensitive, whitespace-tolerant
    comparison against known model aliases.

    Args:
        user_input (str): Raw string from user input.

    Returns:
        str: Lowercase string with hyphens, underscores, and spaces removed.
    """
    return user_input.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def null_matrices(data):
    """
    Build a summary DataFrame of unique value counts and null value counts
    for each column, used to audit data quality after preprocessing.

    Args:
        data (pd.DataFrame): The DataFrame to inspect.

    Returns:
        pd.DataFrame: Two-column summary with unique and null counts per column.
    """
    missing = data.isnull().sum()
    return pd.DataFrame({
        "Unique Value Count": data.nunique(),
        "Null Value Count": missing,
    })


def preprocessing_data(df):
    """
    Clean and prepare the raw DataFrame for feature engineering and modelling.

    Steps performed:
        - Parse release_date into release_year and release_month.
        - Drop irrelevant or leaking columns (regional sales, image URLs, etc.).
        - Remove rows missing developer, release_year, or total_sales.
        - Remove rows where total_sales is zero.
        - Impute missing critic_score using a hierarchy of group means:
            1. Genre + developer mean
            2. Genre + publisher mean
            3. Genre + console mean
            4. Genre mean
            5. Global mean (fallback)
        - Impute missing release_month with the most common month per genre,
          defaulting to November (month 11) if no mode is available.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        pd.DataFrame: Cleaned and imputed DataFrame ready for modelling.
    """
    print("Beginning Data Preprocessing...")

    # Parse release_date and extract year/month components
    df["release_date"] = pd.to_datetime(df["release_date"], dayfirst=True, errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    # Drop columns not needed for modelling
    df = df.drop(
        columns=[
            "img", "other_sales", "na_sales", "jp_sales",
            "pal_sales", "release_date", "last_update"
        ],
        errors="ignore"
    )

    # Remove rows with missing values in required fields and zero-sales entries
    df = df.dropna(subset=["developer", "release_year", "total_sales"])
    df = df[df["total_sales"] != 0]
    df = df.reset_index(drop=True)

    # --- Critic Score Imputation ---
    # Compute group-level mean scores using three levels of granularity
    developer_genre = df.groupby(["genre", "developer"])["critic_score"].transform("mean")
    publisher_genre = df.groupby(["genre", "publisher"])["critic_score"].transform("mean")
    console_genre = df.groupby(["genre", "console"])["critic_score"].transform("mean")

    # Average the three group means to form a combined imputation signal
    combined_score = pd.concat(
        [developer_genre, publisher_genre, console_genre],
        axis=1
    ).mean(axis=1)

    df["critic_score"] = df["critic_score"].fillna(combined_score)

    # Fall back to genre-level mean for any still-missing scores
    if df["critic_score"].isnull().any():
        df["critic_score"] = df["critic_score"].fillna(
            df.groupby("genre")["critic_score"].transform("mean")
        )

    # Final fallback: global mean
    df["critic_score"] = df["critic_score"].fillna(df["critic_score"].mean())
    df["critic_score"] = df["critic_score"].round(1)

    # --- Release Month Imputation ---
    # Use most common month within genre; default to November if mode unavailable
    df["release_month"] = df["release_month"].fillna(
        df.groupby("genre")["release_month"].transform(lambda x: x.mode()[0] if not x.mode().empty else 11)
    ).fillna(11)
    df["release_month"] = df["release_month"].astype(int)

    print("Processed Data Shape:", df.shape)
    print(null_matrices(df))

    return df


def execute_distributions(df):
    """
    Optionally display exploratory heatmap and bar chart distributions.

    Prompts the user to choose from three distribution views:
        - gxp: Genre by Platform (console) heatmap
        - gxs: Genre by Critic Score heatmap
        - gxr: Genre by Release Year heatmap + individual genre bar charts

    The loop continues until the user enters 'q'.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame to visualize.
    """
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

            # Plot individual bar charts showing each genre's frequency by year
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

        # Mask zero-count cells so the heatmap only highlights meaningful data
        mask = ct == 0
        plt.figure(figsize=(28, 20))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Spectral", linewidths=0.5, vmax=500, mask=mask)
        plt.title(title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.show()


def prepare_features_and_labels(df):
    """
    Encode features and bin the target variable into sales tier labels.

    Feature encoding strategy:
        - Low-cardinality categoricals (genre, console, year, month):
          OneHotEncoder with unknown-category handling.
        - High-cardinality categoricals (publisher, developer, title) and
          numeric-like fields (critic_score): TargetEncoder, which encodes
          each value as the mean target observed during training.

    Target binning:
        total_sales is discretized into three ordinal classes:
            1 = High Sales   (> 1M units)
            2 = Medium Sales (0.1M – 1M units)
            3 = Low Sales    (< 0.1M units)

    A stratified 80/20 train-test split is applied to preserve class ratios.

    Args:
        df (pd.DataFrame): Cleaned DataFrame containing all features and target.

    Returns:
        tuple:
            X_raw        (pd.DataFrame): Un-encoded feature matrix (for predictor UI).
            x_train      (np.ndarray):   Encoded training features.
            x_test       (np.ndarray):   Encoded test features.
            y_train      (pd.Series):    Training labels.
            y_test       (pd.Series):    Test labels.
            preprocessor (ColumnTransformer): Fitted encoder for inference-time use.
    """
    y_raw = df["total_sales"]
    X_raw = df.drop(columns=["total_sales"])

    low_cardinality_cols = ["genre", "console", "release_year", "release_month"]
    high_cardinality_cols = ["publisher", "developer", "critic_score", "title"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("low_card", OneHotEncoder(handle_unknown="ignore"), low_cardinality_cols),
            ("high_card", TargetEncoder(), high_cardinality_cols)
        ],
        remainder="passthrough"
    )

    # Bin total_sales into three tiers: High (1), Medium (2), Low (3)
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
    """
    Print a standard evaluation report for a fitted classifier.

    Outputs:
        - Overall accuracy score
        - Per-class precision, recall, F1-score (classification report)
        - Confusion matrix

    Args:
        model:            A fitted sklearn-compatible classifier.
        x_test  (array):  Encoded test feature matrix.
        y_test  (Series): True test labels.
    """
    y_pred = model.predict(x_test)

    print("\n--- Model Evaluation ---")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def compare_models(x_train, x_test, y_train, y_test):
    """
    Train all four classifiers on the same split and compare their performance.

    Models evaluated:
        - Random Forest
        - Decision Tree
        - Logistic Regression
        - Support Vector Machine (RBF kernel)

    Outputs a comparison table of accuracy and training time, identifies the
    best model by accuracy, and prints a macro-average AUC summary.

    Optionally displays a 3-panel ROC curve plot (one panel per sales class,
    using a One-vs-Rest strategy) if the user opts in.

    Args:
        x_train (array):   Encoded training features.
        x_test  (array):   Encoded test features.
        y_train (Series):  Training labels.
        y_test  (Series):  Test labels.

    Returns:
        pd.DataFrame: Results table with columns Model, Accuracy, Training Time.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

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
            probability=True,       # Required for predict_proba and ROC computation
            random_state=sp.RANDOM_STATE
        )
    }

    # Class ordering matches the bin labels: 1=High, 2=Medium, 3=Low
    classes = [1, 2, 3]
    label_names = {1: "High Sales", 2: "Medium Sales", 3: "Low Sales"}

    # Binarize y_test into a (n_samples, n_classes) matrix for OvR ROC computation
    y_test_bin = label_binarize(y_test, classes=classes)

    results = []
    roc_data = {}   # Stores per-model FPR/TPR curves and AUC scores for plotting

    for name, model in models.items():
        start_time = time.time()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)    # Shape: (n_samples, n_classes)

        acc = accuracy_score(y_test, y_pred)
        elapsed_time = time.time() - start_time

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Training Time (sec)": round(elapsed_time, 2)
        })

        # --- Per-Class ROC Computation (One-vs-Rest) ---
        fpr_per_class = {}
        tpr_per_class = {}
        roc_auc_per_class = {}

        # Map each class to its correct probability column; model.classes_ ordering
        # is not guaranteed to be [1, 2, 3], so we look up the index explicitly
        class_order = list(model.classes_)

        for i, cls in enumerate(classes):
            col_idx = class_order.index(cls)
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, col_idx])
            roc_auc_per_class[cls] = auc(fpr, tpr)
            fpr_per_class[cls] = fpr
            tpr_per_class[cls] = tpr

        # Macro-average AUC treats all classes equally regardless of support size
        macro_auc = np.mean(list(roc_auc_per_class.values()))

        roc_data[name] = {
            "fpr": fpr_per_class,
            "tpr": tpr_per_class,
            "auc": roc_auc_per_class,
            "macro_auc": macro_auc
        }

    # --- Print Accuracy Table ---
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison Results ---")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["Accuracy"].idxmax()]
    print("\nBest Model Based on Accuracy:")
    print(f"  {best_row['Model']} with accuracy {best_row['Accuracy']}")

    # --- Macro-Average AUC Summary ---
    # Printed unconditionally so the user always sees AUC even if they skip the plot
    print("\n--- Macro-Average AUC Summary ---")
    for name, data in roc_data.items():
        print(f"  {name:<28}: {data['macro_auc']:.4f}")

    # --- ROC Plot (optional) ---
    show_roc = input("\nWould you like to view the ROC comparison plot? (Y/N): ")
    if show_roc.lower() in ["y", "yes"]:

        # Distinct color per model for clear visual separation on the plot
        model_colors = {
            "Random Forest":          "#2196F3",
            "Decision Tree":          "#FF9800",
            "Logistic Regression":    "#4CAF50",
            "Support Vector Machine": "#E91E63"
        }

        # One subplot per sales class; shared Y-axis for easy cross-panel comparison
        fig, axes = plt.subplots(1, len(classes), figsize=(20, 6), sharey=True)
        fig.suptitle("ROC Curves — One-vs-Rest per Sales Class", fontsize=15, fontweight="bold")

        for col_idx, cls in enumerate(classes):
            ax = axes[col_idx]
            ax.set_title(f"Class: {label_names[cls]}", fontsize=12, fontweight="bold")

            # Diagonal reference line representing a random (no-skill) classifier
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Chance (AUC = 0.50)")

            for name, data in roc_data.items():
                fpr = data["fpr"][cls]
                tpr = data["tpr"][cls]
                roc_auc = data["auc"][cls]
                color = model_colors[name]

                ax.plot(fpr, tpr, lw=2, color=color,
                        label=f"{name} (AUC = {roc_auc:.3f})")

            ax.set_xlabel("False Positive Rate", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

    return results_df


def rf_training_plotting(feature_count, x_train, x_test, y_train, y_test):
    """
    Perform an exhaustive grid search over Random Forest hyperparameters and
    select the best configuration using a composite efficiency score.

    Parameter grid:
        - n_estimators: [feature_count, *2, *5, *10]
        - max_depth:    [3, 5, 10, None]
        - criterion:    ["gini", "entropy"]
        - bootstrap:    [True, False]

    Selection metric:
        effective_score = (accuracy * 10) * log(n_estimators)
                          * log(max(depth, feature_count)) - log(training_time)

        This balances predictive performance, model capacity, and training cost.
        Depth is treated as feature_count when None (unbounded) to avoid log(0).

    Args:
        feature_count (int):   Number of input features; used to scale n_estimators.
        x_train (array):       Encoded training features.
        x_test  (array):       Encoded test features.
        y_train (Series):      Training labels.
        y_test  (Series):      Test labels.

    Returns:
        RandomForestClassifier: The fitted model with the highest efficiency score.
    """
    n_estimators_list = [feature_count, feature_count * 2, feature_count * 5, feature_count * 10]
    depths = [3, 5, 10, 0]      # 0 is used as a sentinel value for max_depth=None
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

        # Run multiple iterations per combination to get a stable average accuracy
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

        # Depth display: N=None (unbounded), X=10, otherwise show numeric value
        print(
            f"Estimators: {n_est:<4} | Criterion: {'g' if c == 'gini' else 'e'} "
            f"| Depth: {'N' if d == 0 else 'X' if d == 10 else d} "
            f"| Bootstrap: {1 if ibt else 0} "
            f"| Avg. Acc: {avg_acc:.4f} | Time: {elapsed_time:.2f} sec"
        )

        # Clamp elapsed time away from zero to avoid log(0) errors
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
    """
    Train a single Decision Tree classifier with default parameters and
    print its evaluation report.

    Args:
        x_train (array):  Encoded training features.
        x_test  (array):  Encoded test features.
        y_train (Series): Training labels.
        y_test  (Series): Test labels.

    Returns:
        DecisionTreeClassifier: The fitted model.
    """
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
    """
    Train a Logistic Regression classifier and print its evaluation report.

    Note: Despite the function name referencing 'linear regression', this uses
    LogisticRegression — a classification model. max_iter=1000 is set to
    ensure convergence on high-dimensional encoded feature sets.

    Args:
        x_train (array):  Encoded training features.
        x_test  (array):  Encoded test features.
        y_train (Series): Training labels.
        y_test  (Series): Test labels.

    Returns:
        LogisticRegression: The fitted model.
    """
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
    """
    Train a Support Vector Machine classifier with an RBF kernel and print
    its evaluation report.

    Parameter notes:
        - kernel="rbf":          Radial Basis Function; handles non-linear boundaries.
        - C=1.0:                 Regularization strength; higher C = less regularization.
        - gamma="scale":         Kernel coefficient set to 1 / (n_features * X.var()).
        - probability=True:      Enables predict_proba via Platt scaling (adds train time).
        - class_weight="balanced": Adjusts weights inversely proportional to class frequency.

    Warning: SVM training time scales as O(n²) to O(n³) with dataset size.
    For large datasets, consider subsampling before calling this function.

    Args:
        x_train (array):  Encoded training features.
        x_test  (array):  Encoded test features.
        y_train (Series): Training labels.
        y_test  (Series): Test labels.

    Returns:
        SVC: The fitted model.
    """
    print("\n--- SVM TRAINING STARTED ---")

    model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=sp.RANDOM_STATE
    )

    model.fit(x_train, y_train)
    print_evaluation(model, x_test, y_test)

    return model


def get_prediction_score(model, encoded_sample):
    """
    Predict the sales class for a single encoded sample and extract the
    probability of that sample falling into the High Sales class (label 1).

    Args:
        model:                   A fitted sklearn-compatible classifier.
        encoded_sample (array):  A single-row encoded feature matrix.

    Returns:
        tuple:
            prediction (int):             Predicted class label (1, 2, or 3).
            high_sales_probability (float): Probability of High Sales; 0 if
                                            the model does not support predict_proba.
    """
    prediction = model.predict(encoded_sample)[0]

    high_sales_probability = 0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(encoded_sample)[0]

        # Locate the column corresponding to class label 1 (High Sales)
        for i, class_label in enumerate(model.classes_):
            if int(class_label) == 1:
                high_sales_probability = probabilities[i]
                break

    return prediction, high_sales_probability


def print_prediction_confidence(model, encoded_sample):
    """
    Print the predicted class probabilities for a single sample across all
    three sales tiers.

    Does nothing if the model does not support predict_proba (e.g., hard-margin SVM).

    Args:
        model:                   A fitted sklearn-compatible classifier.
        encoded_sample (array):  A single-row encoded feature matrix.
    """
    if not hasattr(model, "predict_proba"):
        print("\n  Confidence unavailable for this model.")
        return

    probabilities = model.predict_proba(encoded_sample)[0]

    label_names = {
        3: "Low Sales",
        2: "Medium Sales",
        1: "High Sales"
    }

    print("\n  Confidence:")
    for i, class_label in enumerate(model.classes_):
        class_num = int(class_label)
        print(f"    {label_names[class_num]:<13}: {round(probabilities[i] * 100, 2)}%")


def run_game_recommender(model, preprocessor, resolved):
    """
    Generate "what-if" improvement suggestions by varying key features of the
    resolved game input and observing how predictions change.

    Features varied:
        - Release month: October (10), November (11), December (12)
        - Console:       PS4, PS5, Switch, PC, XOne, XS
        - Critic score:  7.5, 8.0, 8.5, 9.0

    Results are ranked by predicted class (ascending, so High Sales first) then
    by High Sales probability (descending). The top 3 unique suggestions are shown.

    Args:
        model:                      A fitted sklearn-compatible classifier.
        preprocessor (ColumnTransformer): Fitted encoder for transforming samples.
        resolved (dict):            The user's finalized input values.
    """
    print("\n--- Suggested Improvements ---")

    test_variations = []

    # Build all variation tuples: (feature_name, value, modified_resolved_dict)
    for m in [10, 11, 12]:
        temp = resolved.copy()
        temp["release_month"] = m
        test_variations.append(("Release Month", m, temp))

    for c in ["PS4", "PS5", "Switch", "PC", "XOne", "XS"]:
        temp = resolved.copy()
        temp["console"] = c
        test_variations.append(("Console", c, temp))

    for s in [7.5, 8.0, 8.5, 9.0]:
        temp = resolved.copy()
        temp["critic_score"] = s
        test_variations.append(("Critic Score", s, temp))

    recommendations = []

    for feature, value, temp in test_variations:
        temp_df = pd.DataFrame([temp])
        temp_encoded = preprocessor.transform(temp_df)

        temp_prediction, temp_high_probability = get_prediction_score(model, temp_encoded)

        recommendations.append({
            "Feature": feature,
            "Value": value,
            "Prediction": int(temp_prediction),
            "High Sales Probability": temp_high_probability
        })

    recommendations_df = pd.DataFrame(recommendations)

    # Sort: best predicted class first (1=High), then by High Sales probability
    recommendations_df = recommendations_df.sort_values(
        by=["Prediction", "High Sales Probability"],
        ascending=[True, False]
    )

    label_map = {
        3: "Low Sales",
        2: "Medium Sales",
        1: "High Sales"
    }

    print("\n  Best What-If Changes:")

    shown = 0
    used_changes = set()

    for _, row in recommendations_df.iterrows():
        change_name = f"{row['Feature']}:{row['Value']}"

        # Skip duplicate feature-value combinations
        if change_name in used_changes:
            continue

        used_changes.add(change_name)

        if row["High Sales Probability"] > 0:
            print(
                f"    {shown + 1}. Change {row['Feature']} to {row['Value']} "
                f"→ {label_map[int(row['Prediction'])]} "
                f"({round(row['High Sales Probability'] * 100, 2)}% high-sales confidence)"
            )
        else:
            print(
                f"    {shown + 1}. Change {row['Feature']} to {row['Value']} "
                f"→ {label_map[int(row['Prediction'])]}"
            )

        shown += 1

        if shown == 3:
            break


def interactive_sales_prediction(model, preprocessor, X_raw):
    """
    Run an interactive terminal loop allowing the user to predict sales
    categories for custom game inputs.

    The user may leave any field blank to fall back on dataset-derived defaults
    (mode for categorical fields, mean for numeric fields). After each prediction,
    the confidence breakdown and top what-if improvement suggestions are shown.
    The loop continues until the user declines to predict another game.

    Args:
        model:                      A fitted sklearn-compatible classifier.
        preprocessor (ColumnTransformer): Fitted encoder for transforming samples.
        X_raw (pd.DataFrame):       Un-encoded feature matrix used to compute defaults.
    """
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
            """Return user value if provided, otherwise the dataset mode for that field."""
            return user_val if user_val else X_raw[field].mode()[0]

        def resolve_num(user_val, field):
            """Return parsed float if provided, otherwise the dataset mean for that field."""
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

            print_prediction_confidence(model, sample_encoded)
            run_game_recommender(model, preprocessor, resolved)

        except Exception as e:
            print(f"\n  Prediction failed: {e}")

        again = input("\nPredict another game? (Y/N): ")
        if again.lower() not in ["y", "yes"]:
            return


def setup_project_data(df):
    """
    Run the full data preparation pipeline: preprocessing, optional distribution
    display, and feature/label preparation.

    This is the shared setup entry point called by all model-specific functions
    to ensure consistent data handling across model types.

    Args:
        df (pd.DataFrame): Raw input DataFrame from read_file().

    Returns:
        tuple: (X_raw, x_train, x_test, y_train, y_test, preprocessor)
               See prepare_features_and_labels() for full type details.
    """
    preprocessed_df = preprocessing_data(df)
    execute_distributions(preprocessed_df)

    X_raw, x_train, x_test, y_train, y_test, preprocessor = prepare_features_and_labels(preprocessed_df)

    return X_raw, x_train, x_test, y_train, y_test, preprocessor


def random_forest_classifier(df, num_estimators):
    """
    Run the full Random Forest pipeline: data preparation, optional model
    comparison, hyperparameter search, and interactive prediction.

    Args:
        df             (pd.DataFrame): Raw input DataFrame.
        num_estimators (str):          User-specified estimator count or "automatic"
                                       to trigger the grid search.

    Returns:
        str: Completion message.
    """
    X_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    # Optionally run a quick multi-model comparison before the RF grid search
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
    """
    Run the full Decision Tree pipeline: data preparation, training, evaluation,
    and interactive prediction.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        str: Completion message.
    """
    x_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    model = train_decision_tree(x_train, x_test, y_train, y_test)
    interactive_sales_prediction(model, preprocessor, x_raw)

    return "Decision Tree Complete"


def linear_regression_classifier(df):
    """
    Run the full Logistic Regression pipeline: data preparation, training,
    evaluation, and interactive prediction.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        str: Completion message.
    """
    x_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    model = train_logistic_regression(x_train, x_test, y_train, y_test)
    interactive_sales_prediction(model, preprocessor, x_raw)

    return "Logistic Regression Complete"


def svm_classifier(df):
    """
    Run the full SVM pipeline: data preparation, training, evaluation,
    and interactive prediction.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        str: Completion message.
    """
    x_raw, x_train, x_test, y_train, y_test, preprocessor = setup_project_data(df)

    model = train_svm(x_train, x_test, y_train, y_test)
    interactive_sales_prediction(model, preprocessor, x_raw)

    return "SVM Complete"


def compare_all_models(df):
    """
    Run the full model comparison pipeline: data preparation and multi-model
    evaluation with optional ROC plot.

    Args:
        df (pd.DataFrame): Raw input DataFrame.

    Returns:
        str: Completion message.
    """
    _, x_train, x_test, y_train, y_test, _ = setup_project_data(df)

    compare_models(x_train, x_test, y_train, y_test)

    return "Model Comparison Complete"


def get_model_choice(user_input):
    """
    Resolve a raw user string to a canonical model key using MODEL_MAP.

    The input is normalized (lowercased, stripped of spaces/hyphens) before
    matching so that inputs like "Random Forest", "rf", and "random-forest"
    all resolve to "randomforest".

    Args:
        user_input (str): Raw model selection string from the terminal.

    Returns:
        str or None: Canonical model key if matched, None otherwise.
    """
    trimmed_user_input = trim_user_input(user_input)

    for key, variations in MODEL_MAP.items():
        trimmed_variations = [trim_user_input(v) for v in variations]
        if trimmed_user_input in trimmed_variations:
            return key

    return None


def main_prompt():
    """
    Entry point for the interactive CLI. Continuously prompts the user to
    select a model, reads the dataset fresh for each run, and dispatches to
    the appropriate pipeline function.

    Accepted inputs (case-insensitive, hyphen/space/underscore tolerant):
        - Random Forest / rf
        - Decision Tree / dt
        - Logistic Regression / lr
        - SVM / support vector machine / svm
        - Compare / all / cmp / c
        - q (quit)

    Invalid inputs default to the Decision Tree classifier.
    """
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