# Air Quality PBL Repository

This repository is a Python-only project for predicting air quality categories using pollutant concentration and meteorological data.

## Repository Rules Followed

- One main repository
- Three phase folders:
  - `Phase_1`
  - `Phase_2`
  - `Phase_3`
- Each phase folder contains:
  - one Jupyter notebook (`.ipynb`)
  - one improved dataset (`.csv`)
  - one report (`.md`)
- All implementation code is written in Python only

## Project Objective

The project builds a complete machine learning pipeline to classify air quality into:

- `Good`
- `Satisfactory`
- `Moderate`
- `Poor`
- `Very Poor`
- `Severe`

using:

- pollutant measurements
- air sensor values
- meteorological variables such as temperature, humidity, and absolute humidity

## Main Files

- [main.py](</C:/Users/user/OneDrive/Desktop/ai mi air quality project/main.py>) - end-to-end Python pipeline
- [requirements.txt](</C:/Users/user/OneDrive/Desktop/ai mi air quality project/requirements.txt>) - required Python packages
- [AirQualityUCI.csv](</C:/Users/user/OneDrive/Desktop/ai mi air quality project/AirQualityUCI.csv>) - main dataset used in the project

## Phase Structure

### Phase 1

- data understanding
- data cleaning
- missing value treatment
- initial improved dataset

### Phase 2

- feature engineering
- date-time transformation
- lag features
- feature selection preparation

### Phase 3

- model training
- Logistic Regression, KNN, and SVM
- evaluation metrics
- confusion matrix and ROC analysis

## How To Run

```bash
pip install -r requirements.txt
python main.py
```

## What `main.py` Generates

When executed in a Python environment, the script writes:

- `Phase_1/improved_dataset_phase_1.csv`
- `Phase_2/improved_dataset_phase_2.csv`
- `Phase_3/improved_dataset_phase_3.csv`
- `Phase_3/outputs/evaluation_report.json`
- `Phase_3/outputs/evaluation_report.md`
- `Phase_3/outputs/selected_features.json`
- `Phase_3/outputs/confusion_matrix_*.png`
- `Phase_3/outputs/roc_*.png`
- `Phase_3/outputs/best_model.pkl`

