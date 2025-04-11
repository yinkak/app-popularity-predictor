import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_path):
    """Load the enhanced combined dataset."""
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at: {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)
    logging.info(f"Data loaded from {data_path} with shape: {df.shape}")
    logging.info("Initial platform counts BEFORE cleaning:")
    logging.info(df["platform"].value_counts())
    logging.info(f"Available columns: {df.columns.tolist()}")
    return df

def filter_fair_features(df):
    """
    Select only fair features and clean them appropriately:
    - content_rating: fill missing with 'everyone'
    - paid_flag: keep as-is
    - app_age_days: keep as-is
    """
    df = df.copy()
    selected_features = ["content_rating", "paid_flag", "app_age_days"]
    logging.info(f"Using fair features: {selected_features}")
    

    df = df.dropna(subset=["platform"]).copy()
    df["content_rating"] = df["content_rating"].fillna("Everyone")
    df = df.dropna(subset=["app_age_days", "paid_flag"])

    df_clean = df[selected_features + ["platform"]].copy()
    logging.info(f"After filtering, dataset shape: {df_clean.shape}")
    

    logging.info("Class balance:\n" + df_clean['platform'].value_counts().to_string())
    return df_clean, selected_features

def build_preprocessor(numerical_features, categorical_features):
    """Build ColumnTransformer for scaling and encoding."""
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

def main():
    # 1. Load data
    data_path = os.path.join('data', 'combined', 'combined_apps_enhanced.csv')
    df = load_data(data_path)

    # 2. Filter and prepare features
    df_clean, features = filter_fair_features(df)
    print("columns used", df_clean.columns)
    print(df_clean.head())
    X = df_clean[features]
    y = df_clean['platform'].astype(str)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Preprocessing
    numerical_features = ["app_age_days", "paid_flag"]
    categorical_features = ["content_rating"]
    preprocessor = build_preprocessor(numerical_features, categorical_features)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. Train
    pipeline.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

    # 7. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Platform Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("images/platform_confusion_matrix.png")
    plt.close()
    logging.info("Confusion matrix plot saved.")

main()
