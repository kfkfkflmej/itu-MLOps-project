# MLOps Project 2025 - The Deploy Squad

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project is related to the Data Science in Production: MLOps and Software Engineering course. We develop a system that can predict whether a user will become a new customer. The original repository can be found [here](https://github.com/lasselundstenjensen/itu-sdse-project)

![Goal](./references/docs/project-architecture.png)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third-party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── go.mod                           <- Go file that defines the module and required dependencies
│
├── go.sum                           <- Go file that ensures continuity and integrity of dependencies
│
├── pipeline.go                      <- Dagger workflow written in Go
│
├── notebooks          <- Jupyter notebooks. The naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         new_customers_classifier and configuration for tools like black
│
├── references         <- Additional explanatory materials.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── new_customers_classifier   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes new_customers_classifier a Python module
    │
    ├── 01_preprocessing.py   <- Script for preprocessing the raw data for model training 
    │
    ├── 02_model_dev.py             <- Script for training prediction models
    │
    ├── 03_model_selection.py             <- Script for model comparison and selection
    │
    ├── 04_deploy.py             <- Script for deploying the best model
    |
    ├── config.py               <- Store useful variables and configuration
    | 
    ├── utils.py                 <- Helper functions used in the scripts
```

# How to run the code

## Artifact creation

#### Run in a container

#### Local testing

## Inference testing

## Maintaining code quality

## Code releases

# Code decisions and reflections
--------

