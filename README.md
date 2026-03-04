# Sentimental

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https://github.com/babaksoft/sentimental/raw/refs/heads/master/pyproject.toml)
![Static Badge](https://img.shields.io/badge/task-classification-orange)
![Static Badge](https://img.shields.io/badge/framework-sklearn-orange)
![GitHub License](https://img.shields.io/github/license/babaksoft/sentimental)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/babaksoft/sentimental/build.yml)


## Business problem

Using a dataset of user-generated text labelled with coarse-grained mental/emotional categories,
we need a text classification model that can be used in the context of support prioritization and
analytics systems.

**Very important notice**

As dataset labels are **not clinically validated**, we'll treat them strictly as
language patterns and **not medical facts**. Consequently, any system built with
this model should not be used for any of the following purposes :
- Diagnosis
- Clinical decision-making
- Individual risk prediction

## Dataset

- Source: [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- Task: Multi-class text classification
- Samples: 53,043 (before cleaning)
- Features: 2 input features + 1 target


|      Feature    | Description                                                               |
|-----------------|---------------------------------------------------------------------------|
| ID              | Row identifier                                                            |
| statement       | User-generated text (variable length)                                     |
| status          | Mental health state associated with user text (target, 7 distinct values) |


## Data Ingestion & Validation

Initial raw data analysis revealed several quality issues:

- ID column
- Missing data
- Duplicate records
- Conflicting labels
- Class imbalance

Cleaning steps:
- Dropped ID column
- Removed rows with missing data (362): 53,043 → 52,681 
- Removed duplicates (1,588): 52,681 → 51,093
- Removed label conflicts (38): 51,093 → 51,055

The cleaned dataset was split into:
- Train: 70%
- Validation: 15%
- Test: 15%

All dataset splits are:
- Stratified on class labels
- Logged in MLflow
- Versioned and frozen using DVC
