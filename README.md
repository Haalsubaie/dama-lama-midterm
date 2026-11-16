# DAMA/LAMA Prediction Project

## About the Project

This is my midterm project for ML Zoomcamp 2026.

I built a machine learning model to predict if an emergency patient might leave against medical advice (DAMA/LAMA).

I used synthetic data because I cannot share real patient data.

## Dataset

The dataset has 5,000 synthetic patient records.

I used these features:

- Age
- Gender
- Hospital code
- Triage level
- Arrival mode
- Waiting time
- Previous emergency visits
- Registration date

The target is `DAMA_LAMA_Target`:

- 0 = Other discharge
- 1 = DAMA/LAMA

## Data Preparation

First, I checked the data for missing values and duplicates.

Then I handled the missing values and created new features from the registration date.

I split the data into training, validation, and test datasets.

I used DictVectorizer to prepare the features for the models.

## Model Training

I trained three models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

These are my validation results:

| Model | AUC |
|---|---|
| Logistic Regression | 0.6012 |
| Decision Tree | 0.5242 |
| Random Forest | 0.6189 |

Random Forest gave the best validation result, so I selected it.

## Final Result

I tested the final model using the test dataset.

**Test AUC: 0.5658**

The result is not very high. The model needs improvement, but I learned how to train, evaluate, and deploy a machine learning model.

This is a learning project using synthetic data. It should not be used for real medical decisions.

## How to Run the Project

First, install the libraries:

```bash
pip install -r requirements.txt
```

Run the prediction service:

```bash
python predict.py
```

Open another terminal and test it:

```bash
python test.py
```

## Docker

I also used Docker to run the model.

Build the Docker image:

```bash
docker build -t dama-lama-midterm .
```

Run the container:

```bash
docker run --rm -p 9696:9696 dama-lama-midterm
```

Then run `python test.py` in another terminal.

## What I Learned

In this project, I learned how to:

- Explore and clean data
- Create new features
- Train different machine learning models
- Compare models using ROC AUC
- Save a trained model
- Create a Flask API
- Run the model using Docker

## Conclusion

This project helped me understand the steps of building a machine learning project from data preparation to deployment.