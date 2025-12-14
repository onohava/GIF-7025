# Regression Models for Earthquake Magnitude Prediction

The purpose of the project is to train, test, and compare the ability of different regression models to predict the magnitude of an earthquake on the Richter scale, based on attributes like its location, depth, etc. Every model will be trained on the same dataset: [All the Earthquakes Dataset : from 1990-2023](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023/data).

Our goal is to determine whether machine learning can help us identify present and future earthquake patterns, and to find out whether any learning technique is more efficient at this task. This includes measuring the impact of specific parameters on the model's accuracy, or even the impact of the logarithmic distribution of earthquake magnitudes.

## Overview

### Technology Stack

We use the Python programming language and its variety of machine-learning-oriented libraries like Scikit-learn and PyTorch. This enables us to quickly develop working models and try different configurations in our search for an efficient model.

### Project Structure

The project currently supports 4 different models, each one of them separated in its own module. This enables the team to implement custom data processing and cleaning for each model, based on its needs. For example, we might want to restrict a model to only use the 4 most relevant parameters to train on the data, while a more complex model might require 8 to make the most accurate predictions possible.

The currently supported models are:
- Long short-term memory (LSTM): `/lstm_experiments`
- DistilBERT: `/encoder_transformers`
- Multilayer Perceptron: `/standard_neural_network`
- Random Forest: `/random_forest_experiment`

## Prerequisites

The requirements for the project are: 
- A working Python environnement with the basic machine-learning and data science libraries like NumPy, Scikit-learn, PyTorch, etc.
- A dataset with earthquake data
  - As previously mentionned, our models are based on the [All the Earthquakes Dataset : from 1990-2023](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023/data). As it contains million of records spanning more than 30 years, it is too large to be included in the repository. You will need to download the CSV and insert it in the respective model directories, or adjust the models for your needs.
