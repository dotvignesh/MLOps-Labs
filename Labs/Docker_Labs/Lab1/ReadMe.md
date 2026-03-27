# Lab 1: Dockerized Wine Classification

This lab trains a machine learning model inside a Docker container.

The original version used the Iris dataset with a Random Forest model.

## Changes Made

1. I replaced the Iris dataset with the Wine dataset from scikit-learn.
2. I replaced the Random Forest model with a Logistic Regression model.
3. I added `StandardScaler` by using a scikit-learn pipeline.
4. I added model evaluation output with test accuracy and a classification report.
5. I changed the saved model file from `iris_model.pkl` to `wine_model.pkl`.
6. I added a `training_summary.txt` file so the container also saves a readable result summary.

## Project Files

- `dockerfile` builds the Docker image
- `src/main.py` trains and evaluates the model
- `src/requirements.txt` contains the Python dependencies

## How to Run

Open a terminal in the `Lab1` folder and run these commands:

```bash
docker build -t lab1:v1 .
docker run --rm lab1:v1
```

## Expected Output

When the container runs, it will:

1. Load the Wine dataset
2. Split the data into training and test sets
3. Train the Logistic Regression pipeline
4. Print the test accuracy
5. Print the classification report
6. Save the trained model as `wine_model.pkl`
7. Save a result summary as `training_summary.txt`

## Example Docker Commands

```bash
docker images
docker ps -a
```

## Submission Note

This lab shows a complete Docker workflow for a machine learning task:

- build a Docker image
- run the container
- train a model inside the container
- save the trained model and summary output
