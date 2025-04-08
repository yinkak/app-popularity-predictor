# Comparing Proprietary vs. Open-Source Apps: Predicting App Popularity 📲

## Overview

This project compares apps from the Google Play Store (proprietary) with those from F-Droid (open-source) by predicting app popularity based on ratings, downloads, and other app features. The initial focus is on building a robust pipeline that covers data acquisition, cleaning, exploratory data analysis (EDA), feature engineering, and model building (both classification and regression). Future extensions may include further analysis on privacy aspects if needed.

## Project Structure

```
├── data/
│ ├── google_play.csv # Kaggle dataset for Google Play apps
│ └── fdroid.json # JSON data scraped/collected from F-Droid
├── notebooks/
│ ├── 1_EDA.ipynb # Jupyter notebook for exploratory data analysis
│ ├── 2_FeatureEngineering.ipynb # Notebook for creating new features
│ └── 3_ModelBuilding.ipynb # Notebook for training and evaluating models
├── src/
│ ├── data_cleaning.py # Script to clean and preprocess raw data
│ ├── feature_engineering.py # Script to create and transform features
│ ├── model_training.py # Script for model training and evaluation
│ └── utils.py # Helper functions
├── requirements.txt # List of required libraries
└── README.md
```

## Required Libraries

The project is implemented in Python and requires the following libraries:

- **pandas** – for data manipulation and cleaning
- **numpy** – for numerical operations
- **scikit-learn** – for model building, evaluation, and pipeline management
- **matplotlib** and **seaborn** – for visualization
- **requests** – for data collection (if scraping F-Droid data)
- **json** – for parsing JSON data

Install the libraries using:

```bash
pip install -r requirements.txt
```

## How to Run the Project

1. Data Acquisition & Cleaning:
   • Ensure that the raw data files (google_play.csv and fdroid.json) are placed in the data/ directory.
   • Run the data cleaning script:

```bash
python3 src/data_cleaning.py
```

This will process the raw data and output cleaned data files in a designated folder (e.g., data/cleaned/).

2. Exploratory Data Analysis (EDA):

   • Open the notebook notebooks/1_EDA.ipynb in Jupyter Notebook or JupyterLab.

   • Follow the cells to explore the data and generate visualizations (e.g., histograms, box plots, heatmaps).

3. Feature Engineering:

   • Open notebooks/2_FeatureEngineering.ipynb and run the cells to create new features such as Price Tier, Size Category, and Update Frequency.

   • The resulting dataset with new features will be saved for use in modeling.

4. Model Building & Evaluation:

   • Open notebooks/3_ModelBuilding.ipynb to train models.

   • For the classification task, the model predicts whether an app will have a “high” rating (e.g., ≥ 4.0).

   • For the regression task, the model predicts the exact rating or number of installs.

   • Models are built using scikit-learn pipelines; results are displayed using model metrics (F1-score, RMSE, etc.).

Files Produced / Expected Outputs:

• Cleaned Data Files: Processed CSV/JSON files in data/cleaned/.

• EDA Visualizations: Figures (histograms, scatter plots, heatmaps) generated in the EDA notebook.

• Engineered Dataset: A dataset with additional features saved for modeling.

• Model Metrics: Output scores (accuracy, F1, RMSE, etc.) displayed in the Model Building notebook.

• Final Models: Saved models or pipeline objects (if applicable) in a designated models folder (optional).
