# Comparing Proprietary vs. Open-Source Apps: Predicting App Popularity 📲

## Overview

This project compares apps from the Google Play Store (proprietary) with those from F-Droid (open source) by analyzing various features and building predictive models to gauge app popularity (using ratings and installs). The project covers all stages from data acquisition, cleaning, combining, and feature derivation to exploratory analysis and model building. The modeling phase includes both a regression task (predicting app rating) and a classification task (predicting platform membership using Random Forest variants).

Research Questions
• Popularity Prediction: Can we accurately predict app popularity (e.g., ratings) using a set of derived features?
• Platform Differences: Are there systematic differences between open-source (F-Droid) and proprietary (Google Play) apps based on attributes such as reviews, app age, size, and install counts?
• Feature Importance & Insights: Which features most strongly influence app popularity metrics?

# Project Structure

```
CMPT-353-PROJECT/
├── data/
│   ├── cleaned/
│   │   ├── fdroid_cleaned.csv         # Cleaned F-Droid data (converted from JSON)
│   │   └── googleplay_cleaned.csv     # Cleaned Google Play data (from Kaggle)
│   ├── combined/
│   │   ├── combined_apps.csv          # Combined dataset (raw version)
│   │   └── combined_apps_enhanced.csv # Combined dataset with additional derived features
│   └── uncleaned/
│       ├── fdroid.json                # Raw F-Droid JSON data scraped from F-Droid repository
│       └── googleplaystore.csv        # Raw Google Play data downloaded from Kaggle
├── images/
│   ├── feature_importances.png        # Barplot for top 10 feature importances (modeling)
│   ├── platform_confusion_matrix.png  # Confusion matrix for platform classification
│   └── platform_confusion_matrix_fair.png # Alternative confusion matrix (if used)
├── notebooks/
│   ├── Feature_Derivation.ipynb       # Notebook for deriving additional features
│   └── Visualize_Trends.ipynb         # Notebook for exploratory visualizations
├── src/
│   ├── acquire_fdroid.py              # Script to download F-Droid data via API
│   ├── acquire_googleplay.py          # Script to download Google Play data from Kaggle
│   ├── clean_fdroid.py                # Script to clean and process F-Droid data
│   ├── clean_googleplay.py            # Script to clean and process Google Play data
│   ├── combine_datasets.py            # Script to merge FDroid and Google Play datasets
│   ├── build_model.py                 # Script to build and evaluate the regression model
│   └── platform_classifier_RandomForests.py  # Script for platform classification using Random Forests
├── README.md                          # This file
├── .gitignore                         # Files and directories to ignore in Git
└── requirements.txt                   # Python package dependencies

```

## Required Libraries

The project is implemented in Python 3.x. The main packages required include:

• pandas – Data manipulation and cleaning

• numpy – Numerical operations

• matplotlib and seaborn – Data visualization

• scikit-learn – Model building, preprocessing, and evaluation

• requests – HTTP requests for data acquisition

• kaggle – To interact with the Kaggle API for data downloads

• jupyter – For running the notebooks

Install the libraries using:

```bash
pip install -r requirements.txt
```

## Setup and Execution

As the files required to run the scripts are all included in the Github repo, you will have them locally once cloned, thus remove the need to acquire data or use inputs. However, the instructions are listed as though you are creating the project from scratch, and in the case of wanting to do further testing, you can use your own Kaggle API key to test different datasets.

1. Data Acquisition:

• F-Droid Data:

Run the script to download F-Droid data:

```bash
python3 src/acquire_fdroid.py
```

This will download the raw fdroid.json file into the data/uncleaned/ directory.

• Google Play Data:

Run the script to download the Google Play Store dataset from Kaggle:

```bash
python3 src/acquire_googleplay.py
```

Make sure you have a valid kaggle.json in the ~/.kaggle/ directory or set the environment variables accordingly.

The raw Google Play data will be saved in data/uncleaned/.

2.  Data Cleaning:

    • F-Droid Cleaning:

Process the raw JSON to produce cleaned CSV data:

```bash
python3 src/clean_fdroid.py
```

This outputs fdroid_cleaned.csv in data/cleaned/.

• Google Play Cleaning:

Process the Kaggle CSV dataset:

```bash
python3 src/clean_googleplay.py
```

This outputs googleplay_cleaned.csv in data/cleaned/.

3. Combining Datasets:

Merge both cleaned datasets into one combined dataset with derived features:

```bash
python3 src/combine_datasets.py
```

This creates one file:

• combined_apps.csv (merged with basic comparative organization)

4. Exploratory Data Analysis (EDA):

Open and run the following Jupyter notebooks for additional exploratory analysis and further feature derivation:

• Visualization Notebook:

```bash
jupyter notebook notebooks/Visualize_Trends.ipynb
```

This notebook generates visualizations (e.g., distributions, correlations, trends) for further data insights.

5. Feature Engineering:

• Feature Derivation Notebook:

```bash
jupyter notebook notebooks/Feature_Derivation.ipynb
```

This notebook loads combined_apps_enhanced.csv, derives new features (such as app age, binned installs, and flags), and saves the enhanced dataset.

• The resulting dataset with new features will be saved for use in modeling.

6. Model Building & Evaluation:

• Regression Model (Predicting Ratings):

Run the regression script that uses a Random Forest pipeline with GridSearchCV:

```bash
python3 src/build_model.py
```

The script prints best hyperparameters, test set R² and RMSE metrics, and saves a feature importance barplot in the images/ directory.

• Platform Classification Model:

Run the script to build a Random Forest classifier for predicting the platform (F-Droid vs. Google Play):

```bash
python3 src/platform_classifier_RandomForests.py
```

This outputs a classification report and saves the confusion matrix plot in the images/ directory.

Expected Outputs:

• Data Files:

- Cleaned data stored in data/cleaned/ and data/combined/ directories.

• Notebooks:

- Visualizations and derived feature datasets are produced from the notebooks in the notebooks/ folder.

• Model Metrics and Artifacts:

- Regression model evaluation metrics (R², RMSE) printed to console.
- Feature importances plot saved as images/feature_importances.png.
- Classification report and confusion matrix plot saved as image files in the images/ directory.

Additional Notes:

• Make sure to update the kaggle.json file in your ~/.kaggle folder if you experience authentication issues with the Kaggle API.

• You may need to adjust column names or feature lists in the code if the format of the input data changes.

• All scripts assume that the project is run from the project root directory (i.e., CMPT-353-PROJECT/).

This README provides detailed documentation of code, instructions for running the scripts, dependencies, and the expected file outputs. It ensures reproducibility and helps users—and evaluators—understand the flow of the project and the methods used to meet the assignment requirements.
