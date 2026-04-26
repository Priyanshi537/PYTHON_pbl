from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "AirQualityUCI.csv"
PHASE_1_DIR = BASE_DIR / "Phase_1"
PHASE_2_DIR = BASE_DIR / "Phase_2"
PHASE_3_DIR = BASE_DIR / "Phase_3"
OUTPUT_DIR = PHASE_3_DIR / "outputs"

AQI_CLASSES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
POLLUTANT_COLUMNS = [
    "CO(GT)",
    "NMHC(GT)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
]
LAG_COLUMNS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"]


def ensure_project_dirs() -> None:
    PHASE_1_DIR.mkdir(exist_ok=True)
    PHASE_2_DIR.mkdir(exist_ok=True)
    PHASE_3_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def save_phase_dataset(df: pd.DataFrame, destination: Path) -> None:
    export_df = df.copy()
    if "datetime" in export_df.columns:
        export_df["datetime"] = export_df["datetime"].astype(str)
    export_df.to_csv(destination, index=False)


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH, sep=";")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [column.strip() for column in df.columns]

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.replace(",", ".", regex=False)

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    numeric_columns = [column for column in df.columns if column not in {"Date", "Time", "datetime"}]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    df[numeric_columns] = df[numeric_columns].replace(-200, np.nan)
    return df


def build_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    scaled_pollutants = pd.DataFrame(index=df.index)

    for column in POLLUTANT_COLUMNS:
        low = df[column].quantile(0.10)
        high = df[column].quantile(0.90)
        denominator = high - low if pd.notna(high) and pd.notna(low) and high != low else 1.0
        scaled_pollutants[column] = ((df[column] - low) / denominator).clip(0, 1)

    df["pollution_score"] = scaled_pollutants.mean(axis=1, skipna=True)

    ranked_scores = df["pollution_score"].rank(method="first")
    df["current_aqi_category"] = pd.qcut(
        ranked_scores,
        q=len(AQI_CLASSES),
        labels=AQI_CLASSES,
    )

    next_is_hourly = df["datetime"].shift(-1) - df["datetime"] == pd.Timedelta(hours=1)
    df["target_aqi_category"] = df["current_aqi_category"].shift(-1)
    df = df[next_is_hourly].copy()
    df = df.dropna(subset=["target_aqi_category"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    featured["hour"] = featured["datetime"].dt.hour
    featured["day"] = featured["datetime"].dt.day
    featured["month"] = featured["datetime"].dt.month
    featured["weekday"] = featured["datetime"].dt.weekday
    featured["is_weekend"] = featured["weekday"].isin([5, 6]).astype(int)
    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24)
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12)
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12)
    featured["temp_humidity_interaction"] = featured["T"] * featured["RH"]

    for column in LAG_COLUMNS:
        featured[f"lag1_{column}"] = featured[column].shift(1)

    previous_gap_is_hourly = featured["datetime"] - featured["datetime"].shift(1) == pd.Timedelta(hours=1)
    lag_columns = [f"lag1_{column}" for column in LAG_COLUMNS]
    featured.loc[~previous_gap_is_hourly, lag_columns] = np.nan
    return featured


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "Date",
        "Time",
        "datetime",
        "pollution_score",
        "current_aqi_category",
        "target_aqi_category",
    }
    return [column for column in df.columns if column not in excluded]


def choose_features(X_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str], k_best: int = 15) -> tuple[list[str], SelectKBest]:
    selector = SelectKBest(score_func=f_classif, k=min(k_best, len(feature_names)))
    selector.fit(X_train, y_train)

    score_frame = pd.DataFrame({"feature": feature_names, "score": selector.scores_}).fillna(0.0)
    score_frame = score_frame.sort_values("score", ascending=False).reset_index(drop=True)

    selected = score_frame.head(min(12, len(score_frame)))["feature"].tolist()
    weather_candidates = [item for item in ["T", "RH", "AH", "lag1_T", "lag1_RH", "lag1_AH"] if item in score_frame["feature"].values]
    time_candidates = [item for item in ["hour", "hour_sin", "hour_cos", "weekday", "month"] if item in score_frame["feature"].values]

    for feature in weather_candidates:
        weather_selected = [item for item in selected if item in weather_candidates]
        if feature not in selected and len(weather_selected) < 2:
            selected.append(feature)

    for feature in time_candidates:
        if feature not in selected:
            selected.append(feature)
            break

    return selected, selector


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000)),
            ]
        ),
        "knn": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=9)),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True)),
            ]
        ),
    }


def save_confusion_matrix(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=AQI_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=AQI_CLASSES).plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confusion_matrix_{model_name}.png", dpi=180)
    plt.close(fig)


def save_roc_curve(model_name: str, y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_bin = label_binarize(y_true, classes=AQI_CLASSES)
    per_class_auc: dict[str, float] = {}

    fig, ax = plt.subplots(figsize=(8, 6))
    for index, label in enumerate(AQI_CLASSES):
        if y_bin[:, index].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, index], y_prob[:, index])
        class_auc = roc_auc_score(y_bin[:, index], y_prob[:, index])
        per_class_auc[label] = float(class_auc)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={class_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"roc_{model_name}.png", dpi=180)
    plt.close(fig)
    return per_class_auc


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    save_confusion_matrix(model_name, y_test, y_pred)
    per_class_auc = save_roc_curve(model_name, y_test, y_prob)

    macro_auc = roc_auc_score(
        label_binarize(y_test, classes=AQI_CLASSES),
        y_prob,
        average="macro",
        multi_class="ovr",
    )

    return {
        "name": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": float(macro_auc),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=AQI_CLASSES).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "per_class_auc": per_class_auc,
        "model": model,
    }


def write_markdown_report(report: dict) -> None:
    lines = [
        "# Air Quality Category Prediction Report",
        "",
        f"- Dataset: `{report['dataset']['source']}`",
        f"- Total modeled rows: {report['dataset']['total_rows']}",
        f"- Training rows: {report['dataset']['train_rows']}",
        f"- Test rows: {report['dataset']['test_rows']}",
        f"- Selected features: {', '.join(report['preprocessing']['selected_features'])}",
        "",
        "## Target Distribution",
        "",
    ]

    for label, count in report["dataset"]["target_distribution"].items():
        lines.append(f"- {label}: {count}")

    for model in report["models"]:
        lines.extend(
            [
                "",
                f"## {model['name']}",
                "",
                f"- Accuracy: {model['accuracy']:.4f}",
                f"- Macro Precision: {model['precision_macro']:.4f}",
                f"- Macro Recall: {model['recall_macro']:.4f}",
                f"- Macro F1: {model['f1_macro']:.4f}",
                f"- Macro ROC AUC: {model['roc_auc_macro_ovr']:.4f}",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )

        for label in AQI_CLASSES:
            class_metrics = model["classification_report"].get(label, {})
            lines.append(
                f"| {label} | {class_metrics.get('precision', 0):.4f} | {class_metrics.get('recall', 0):.4f} | {class_metrics.get('f1-score', 0):.4f} | {int(class_metrics.get('support', 0))} |"
            )

        lines.extend(["", "Confusion Matrix:", "", "```text"])
        for row in model["confusion_matrix"]:
            lines.append(" ".join(str(value) for value in row))
        lines.append("```")

    (OUTPUT_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_project_dirs()

    df = load_and_clean_data()
    save_phase_dataset(df, PHASE_1_DIR / "improved_dataset_phase_1.csv")

    df = build_target_labels(df)
    df = engineer_features(df)
    save_phase_dataset(df, PHASE_2_DIR / "improved_dataset_phase_2.csv")

    feature_columns = select_feature_columns(df)
    model_df = df[feature_columns + ["target_aqi_category", "datetime"]].copy()
    model_df = model_df.sort_values("datetime").reset_index(drop=True)
    save_phase_dataset(model_df, PHASE_3_DIR / "improved_dataset_phase_3.csv")

    split_index = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    X_train_raw = train_df[feature_columns]
    y_train = train_df["target_aqi_category"].astype(str)
    X_test_raw = test_df[feature_columns]
    y_test = test_df["target_aqi_category"].astype(str)

    selector_imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        selector_imputer.fit_transform(X_train_raw),
        columns=feature_columns,
        index=X_train_raw.index,
    )

    selected_features, feature_selector = choose_features(X_train_imputed, y_train, feature_columns)
    X_train = X_train_raw[selected_features]
    X_test = X_test_raw[selected_features]

    model_results = []
    for model_name, model in build_models().items():
        model_results.append(evaluate_model(model_name, model, X_train, y_train, X_test, y_test))

    best_model_result = max(model_results, key=lambda item: item["f1_macro"])
    best_model = best_model_result["model"]

    with (OUTPUT_DIR / "best_model.pkl").open("wb") as file:
        pickle.dump(
            {
                "model": best_model,
                "selected_features": selected_features,
                "classes": AQI_CLASSES,
            },
            file,
        )

    target_distribution = (
        model_df["target_aqi_category"]
        .astype(str)
        .value_counts()
        .reindex(AQI_CLASSES, fill_value=0)
        .to_dict()
    )

    feature_scores = pd.DataFrame(
        {
            "feature": feature_columns,
            "score": feature_selector.scores_,
        }
    ).fillna(0.0)
    feature_scores = feature_scores.sort_values("score", ascending=False)

    report = {
        "dataset": {
            "source": DATASET_PATH.name,
            "total_rows": int(len(model_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "target_distribution": {key: int(value) for key, value in target_distribution.items()},
        },
        "preprocessing": {
            "missing_value_treatment": "Replace -200 sentinel values with NaN, then apply median imputation",
            "normalization": "StandardScaler on selected features",
            "feature_selection_method": "ANOVA F-score ranking with meteorological feature coverage",
            "selected_features": selected_features,
            "top_feature_scores": feature_scores.head(20).to_dict(orient="records"),
            "target_definition": "Next-hour AQI-style severity category derived from quantile-binned pollutant score",
        },
        "models": [],
        "best_model": best_model_result["name"],
    }

    for result in model_results:
        sanitized = {key: value for key, value in result.items() if key != "model"}
        report["models"].append(sanitized)

    (OUTPUT_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (OUTPUT_DIR / "selected_features.json").write_text(
        json.dumps(report["preprocessing"], indent=2, default=str),
        encoding="utf-8",
    )
    write_markdown_report(report)

    print("Python air quality classification pipeline is ready.")
    print(f"Rows used for modeling: {len(model_df)}")
    print(f"Selected features: {', '.join(selected_features)}")
    print(f"Best model by macro F1: {report['best_model']}")


if __name__ == "__main__":
    main()

