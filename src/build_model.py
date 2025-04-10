#!/usr/bin/env python3
"""
build_model.py

This script builds and evaluates a regression model to predict app ratings using the enhanced,
combined dataset ("data/combined/combined_apps_enhanced.csv"). The model is based on a Random Forest
Regressor. The pipeline includes preprocessing steps that scale numerical features and one-hot encode
categorical features. Hyperparameters are tuned using GridSearchCV, and the model is evaluated using
R² and RMSE metrics. Finally, the script extracts and saves a barplot of the top 10 feature importances.
"""

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(data_path):
    """Load the enhanced combined dataset."""
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at: {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)
    logging.info(f"Data loaded from {data_path} with shape: {df.shape}")
    return df


def preprocess_data(df):
    """
    Preprocess the data and select features for modeling.
    
    Drops rows missing the target variable ('rating') and selects a set of features.
    """
    # Drop rows missing the target rating
    df = df.dropna(subset=['rating'])
    logging.info(f"After dropping missing ratings, data shape: {df.shape}")

    # Select features for modeling. You can adjust this list based on further analysis.
    features = [
        'app_age_days',  # Derived app age in days since last update
        'reviews',       # Number of reviews
        'size_mb',       # App size in MB
        'installs_clean',# Cleaned install numbers
        'paid_flag',     # Binary flag (0: Free, 1: Paid)
        'installs_bin',  # Binned install categories
        'category',      # App category (categorical)
        'platform',      # Platform identifier (e.g., FDroid, Google Play)
        'last_updated_year'  # Extracted year from the last update
    ]
    target = 'rating'

    # Drop rows that have missing values in selected features
    df_model = df[features + [target]].dropna()
    logging.info(f"After selecting features and dropping missing values, data shape: {df_model.shape}")
    return df_model, features, target


def build_preprocessor(numerical_features, categorical_features):
    """Build a ColumnTransformer to preprocess numerical and categorical features."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def build_pipeline(preprocessor):
    """Construct a modeling pipeline with the preprocessor and a Random Forest regressor."""
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    return pipeline


def main():
    # Define the path to the enhanced combined dataset (relative to project root)
    data_path = os.path.join('data', 'combined', 'combined_apps_enhanced.csv')
    df = load_data(data_path)

    # Preprocess the data to select features for modeling
    df_model, features, target = preprocess_data(df)

    # Split the dataset into features (X) and target (y)
    X = df_model.drop(target, axis=1)
    y = df_model[target]

    # Split into training and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    logging.info(f"Training set shape: {X_train.shape}")
    logging.info(f"Test set shape: {X_test.shape}")

    # Define numerical and categorical features (ensure these match the selections in preprocess_data)
    numerical_features = ['app_age_days', 'reviews', 'size_mb', 'installs_clean', 'paid_flag', 'last_updated_year']
    categorical_features = ['installs_bin', 'category', 'platform']

    # Build the preprocessor and the pipeline
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    pipeline = build_pipeline(preprocessor)

    # Define hyperparameter grid for the RandomForestRegressor
    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [None, 10, 20]
    }

    # Configure GridSearchCV to tune hyperparameters with 5-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best hyperparameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation R^2 score: {grid_search.best_score_:.4f}")

    # Evaluate the best model on the test set
    y_pred = grid_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"Test set R^2: {r2:.4f}")
    logging.info(f"Test set RMSE: {rmse:.4f}")

    # Extract feature importances from the best model's regressor
    best_model = grid_search.best_estimator_
    regressor = best_model.named_steps['regressor']
    importances = regressor.feature_importances_

    # Retrieve one-hot encoded feature names from the preprocessor for categorical features
    cat_feature_names = list(best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
    all_features = numerical_features + cat_feature_names

    # Create a DataFrame for feature importances and sort them
    feat_importances = pd.DataFrame({'feature': all_features, 'importance': importances})
    feat_importances = feat_importances.sort_values(by='importance', ascending=False)
    logging.info("Top 10 Feature Importances:")
    print(feat_importances.head(10))

    # Plot and save the top 10 feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feat_importances.head(10))
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plot_path = os.path.join('data', 'combined', 'feature_importances.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Feature importances plot saved to {plot_path}")


if __name__ == "__main__":
    main()