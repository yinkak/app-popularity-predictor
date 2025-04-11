import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(data_path):
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at: {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)
    logging.info(f"Data loaded from {data_path} with shape: {df.shape}")
    return df

def preprocess_data(df, include_price=True):
    df = df.dropna(subset=["rating"])
    logging.info(f"After dropping missing ratings, data shape: {df.shape}")

    df['last_updated_date'] = pd.to_datetime(df['last_updated_date'], errors='coerce')
    today = pd.Timestamp.today()
    df['app_age_days'] = (today - df['last_updated_date']).dt.days

    bins = [0, 10000, 100000, 1000000, np.inf]
    labels = ['<10K', '10K-100K', '100K-1M', '1M+']
    df['installs_bin'] = pd.cut(df['installs_clean'], bins=bins, labels=labels)

    df['log_reviews'] = np.log1p(df['reviews'])
    df['log_installs'] = np.log1p(df['installs_clean'])

    features = ["app_age_days", "log_reviews", "size_mb", "log_installs"]
    if include_price:
        features.append("price_clean")
    features += ["installs_bin", "category"]

    target = "rating"
    df_model = df[features + [target]].dropna()
    logging.info(f"Data shape with{'out' if not include_price else ''} price: {df_model.shape}")
    return df_model, features, target

def build_preprocessor(numerical_features, categorical_features):
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])
    return preprocessor

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"{name} R^2: {r2:.4f}, RMSE: {rmse:.4f}")
    return y_pred, r2, rmse

def plot_predictions(y_true, y_pred, title, filename):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([1, 5], [1, 5], linestyle='--', color='red')
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_feature_importance(model, preprocessor, numerical_features, categorical_features, filename):
    regressor = model.named_steps['regressor']
    importances = regressor.feature_importances_
    cat_feature_names = list(preprocessor.transformers_[1][1].get_feature_names_out(categorical_features))
    all_features = numerical_features + cat_feature_names
    feat_importances = pd.DataFrame({"feature": all_features, "importance": importances})
    feat_importances = feat_importances.sort_values(by="importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feat_importances.head(10))
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Feature importance plot saved to {filename}")

def run_pipeline(include_price):
    label = "with_price" if include_price else "without_price"
    df = load_data(os.path.join("data", "cleaned", "googleplay_cleaned.csv"))
    df_model, features, target = preprocess_data(df, include_price=include_price)

    X = df_model.drop(target, axis=1)
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numerical_features = [f for f in ["app_age_days", "log_reviews", "size_mb", "log_installs", "price_clean"] if f in X.columns]
    categorical_features = ["installs_bin", "category"]

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    rf_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))])
    rf_pipeline.fit(X_train, y_train)

    y_pred, r2, rmse = evaluate_model(f"RandomForest ({label})", rf_pipeline, X_test, y_test)
    plot_predictions(y_test, y_pred, f"Predicted vs Actual Ratings ({label})", f"images/pred_vs_actual_{label}.png")

    if include_price:
        plot_feature_importance(rf_pipeline, preprocessor, numerical_features, categorical_features, f"images/feature_importance_{label}.png")

    return r2, rmse

def main():
    r2_with, rmse_with = run_pipeline(include_price=True)
    r2_without, rmse_without = run_pipeline(include_price=False)
    logging.info(f"\nComparison Summary:\nWith Price - R^2: {r2_with:.4f}, RMSE: {rmse_with:.4f}\nWithout Price - R^2: {r2_without:.4f}, RMSE: {rmse_without:.4f}")

if __name__ == "__main__":
    main()