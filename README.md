# Planetary Defense Analytics: Classifying, Predicting and Clustering Asteroids

## Project Overview

Planetary Defense Analytics is a machine-learning project focused on analyzing asteroid data to explore potentially hazardous asteroids, predict asteroid diameters, and identify asteroid groups using clustering techniques.

The project includes a Streamlit web application that allows users to interact with trained machine-learning models and view model evaluation results.

## Objectives

* Classify potentially hazardous asteroids (PHAs).
* Predict asteroid diameter using regression models.
* Group asteroids using K-Means clustering.
* Explore DBSCAN clustering.
* Display model evaluation results.
* Provide an interactive interface using Streamlit.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

## Application Features

### 1. Overview

Provides an introduction to the project and its purpose.

### 2. PHA Classification

Uses a saved classification model to predict whether an asteroid is potentially hazardous based on the application's input features.

### 3. Diameter Prediction

Predicts asteroid diameter using trained regression models.

### 4. K-Means Clustering

Uses a saved K-Means model to group asteroids based on the selected features.

### 5. Model Evaluation

Displays saved model evaluation results from `model_evaluation.json`.

## Saved Models

The `planetary_defense_models/` folder contains the saved machine-learning artifacts:

* `pha_classifier.pkl`
* `diameter_regressor.pkl`
* `diameter_regressor_log.pkl`
* `kmeans_model.pkl`
* `dbscan_model.pkl`
* `cluster_imputer.pkl`
* `cluster_scaler.pkl`

## Project Structure

```text
planetary_defense_analytics/
├── planetary_defense_models/
│   ├── cluster_imputer.pkl
│   ├── cluster_scaler.pkl
│   ├── dbscan_model.pkl
│   ├── diameter_regressor_log.pkl
│   ├── diameter_regressor.pkl
│   ├── kmeans_model.pkl
│   └── pha_classifier.pkl
├── app.py
├── model_evaluation.json
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit application

```bash
python -m streamlit run app.py
```

### 3. Open the application

Open the local URL displayed in the terminal after Streamlit starts.

## Dataset

The original asteroid dataset is not included in this repository to avoid uploading a large data file. The application uses saved model files and evaluation results.

## Future Improvements

* Improve PHA classification recall.
* Add model explainability features.
* Enhance visualizations and model comparisons.
* Expand clustering analysis.
* Improve the application's interface.

## Disclaimer

This project is intended for educational and analytical purposes only. Its predictions are not official asteroid threat assessments and must not be used as a substitute for professional scientific analysis.
