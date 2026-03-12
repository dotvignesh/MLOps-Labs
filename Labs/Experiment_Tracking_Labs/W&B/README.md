# W&B Experiment Tracking Labs Submission

In this submission, I completed both W&B labs with my own changes so they are not identical to the original repo.

- In Lab1.ipynb, I use the Wine dataset and a Random Forest classifier instead of the original dataset and XGBoost model.
- In Lab2.ipynb, I use the MNIST dataset and a custom CNN instead of Fashion-MNIST.

The main goal in both notebooks is to practice experiment tracking with Weights & Biases by logging runs, metrics, visual results, checkpoints, and model outputs.

## How to run

1. Open the project folder:

```bash
cd "/Users/vigneshraja/Documents/NEU Notes/MLOps/MLOps-Labs/Labs/Experiment_Tracking_Labs/W&B"
```

2. Create a virtual environment with `uv`:

```bash
uv venv .venv
```

3. Activate the environment:

```bash
source .venv/bin/activate
```

4. Install the required packages:

```bash
uv pip install notebook jupyterlab wandb scikit-learn pandas numpy tensorflow
```

5. Start Jupyter:

```bash
jupyter notebook
```

6. Open and run these notebooks:

- Lab1.ipynb
- Lab2.ipynb

7. When the notebook asks for W&B access, log in with my Weights & Biases account.

After that, each notebook will create its own tracked run in W&B.
