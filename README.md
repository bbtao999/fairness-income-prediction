# Fairness in Income Prediction

This project compares two in-processing debiasing approaches on the UCI Adult income dataset:

- adversarial debiasing with a PyTorch neural network
- LightGBM with a custom fairness-penalized objective

The main result summary is in [RESULTS.md](RESULTS.md).

## Setup

This project uses `uv` for Python dependency management.

```bash
uv sync --locked
```

The raw dataset is not included in this repository. Download the UCI Adult data into `data/adult.data`:
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

