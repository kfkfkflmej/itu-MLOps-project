# GLOBAL VARS
PROJECT_NAME = new_customers_classifier
PYTHON_INTERPRETER = python
PIP = pip

#COMMANDS
.PHONY: help install clean data lint format test train tune deploy

## Display help text
help:
	@echo "$(PROJECT_NAME) - Customer Classifier Makefile"
	@echo "------------------------------------------------"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# DEPENDENCIES

## Install dependencies from pyproject.toml
install:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -e .[dev]
	

# DATA AND DVC
## Pull latest version of raw_data.csv from remote storage
data-pull:
	dvc pull data/raw/raw_data.csv.dvc

data-add:
	dvc add data/raw/raw_data.csv
	git add data/raw/raw_data.csv.dvc .gitignore

# CODE QUALITY AND TESTING

## Format code (Black)
format:
	black $(PROJECT_NAME) tests

## Lint code (Flake8)
lint:
	flake8 $(PROJECT_NAME)

## Run pipeline test (verbose)
test:
	dagger run go test -v

# MODEL PIPELINE SCRIPTS
## Run Preprocessing script
preprocess:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).preprocessing

## Run Model Selection script
tune:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).model_selection

## Run Training script
train:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).model_dev

## Run Deployment script
deploy:
	$(PYTHON_INTERPRETER) -m $(PROJECT_NAME).deploy

# CLEANUP
## Delete compiled python files and artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf data/interim/* data/processed/* models/*
	touch data/interim/.gitkeep data/processed/.gitkeep models/.gitkeep