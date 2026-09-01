# [Student Job Profile Suggester (Live Demo)](https://student-job-profile-suggester.streamlit.app/)

A machine learning system that predicts a suitable job profile for a student — e.g. Software Developer, Data Scientist, Systems Engineer — based on their academic scores and technical skills. Built as a term paper project (Machine Learning, Lovely Professional University) and deployed as a live Streamlit web app.

## Overview

Given a student's scores across core CS subjects (DSA, DBMS, OS, CN, Mathematics, Aptitude, Communication, Problem Solving, Creativity, Hackathon participation) plus their top two technical skills, the model predicts the job profile they're best suited for. Four classifiers — Decision Tree, Random Forest, Gradient Boosting, and XGBoost — were trained and compared, with hyperparameters tuned via grid search and Bayesian optimization. **XGBoost was selected as the final model, achieving 97% accuracy** on the held-out test set.

## Features
- Predicts suitable job profiles based on a student's academic scores and skillset
- Compares four ML models (Decision Tree, Random Forest, Gradient Boosting, XGBoost) with hyperparameter tuning
- Real-time predictions through an interactive Streamlit interface
- Trained, evaluated, and deployed end to end — from raw data to a live public app

## Technologies Used
Python · Pandas · NumPy · Scikit-learn · XGBoost · Joblib · Matplotlib · Seaborn · Streamlit

## Dataset
Student records containing scores in DSA, DBMS, OS, CN, Mathematics, Aptitude, Communication, Problem Solving, and Creativity, along with hackathon participation and top two technical skills, with job profile as the target variable.

## Models
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier (final model — 97% accuracy)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run app.py
```

Or try the live app directly: https://student-job-profile-suggester.streamlit.app/
