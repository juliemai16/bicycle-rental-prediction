# Bicycle Rental Prediction Project

## Introduction

This project aims to predict the number of bicycles rented per hour based on various environmental and temporal factors. The dataset is sourced from the Kaggle competition "Proton X TF 09 - Bài toán dự đoán xe đạp".

## Project Overview

In this project, we develop and evaluate multiple machine-learning models to accurately forecast bicycle rental demand. The models include:
- TensorFlow neural network
- LightGBM
- Random Forest
- Gradient Boosting
- XGBoost

## Motivation

Understanding and predicting bicycle rental patterns can help optimize bike-sharing systems and improve user experience.

## Setup

### Environment Setup

To set up your environment, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/juliemai16/bicycle-rental-prediction.git
    cd bicycle-rental-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Data

### Data Collection

The dataset is provided by the Kaggle competition and can be downloaded from [this link](https://www.kaggle.com/competitions/proton-x-tf-09-bai-toan-du-doan-xe-dap/data). It includes various features such as season, year, month, hour, weather conditions, temperature, and humidity.

### Data Preprocessing

Data preprocessing steps include:
- Handling missing values
- Encoding categorical variables
- Normalizing numerical features
- Splitting the data into training and validation sets

### Exploratory Data Analysis (EDA)

EDA involves:
- Understanding the distribution of the target variable
- Analyzing relationships between features and the target
- Identifying patterns and anomalies in the data

### Data Visualization

Data visualization techniques are used to:
- Visualize the distribution of bicycle rentals
- Explore temporal trends (e.g., rentals by hour, day, month)
- Examine the impact of weather conditions on rentals

### Statistical Analysis

Statistical analysis is performed to:
- Identify significant features
- Understand correlations between features and the target variable

## Modeling

### Model Architecture

We implement and compare several machine learning models, including:
- A neural network using TensorFlow
- Gradient boosting models (LightGBM, XGBoost)
- Random Forest

### Model Training

Models are trained using the preprocessed data. Key steps include:
- Splitting data into training and validation sets
- Hyperparameter tuning
- Early stopping to prevent overfitting

### Evaluation Metrics

Model performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Deployment

### Deployment Instructions

To deploy the best model for making predictions on new data:

1. Ensure the preprocessed data is available.
2. Load the saved model and preprocessor.
3. Use the model to make predictions on the new data.


## Results

### Performance Evaluation

Model performance is summarized with metrics and visualizations. Key insights include:
- Comparison of model performance
- Analysis of prediction accuracy

### Discussion

Discuss the results, including:
- Model strengths and weaknesses
- Potential improvements
- Implications for real-world applications

## Contributing

### Contribution Guidelines

To contribute to this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [Kaggle Competition: Proton X TF 09 - Bài toán dự đoán xe đạp](https://www.kaggle.com/competitions/proton-x-tf-09-bai-toan-du-doan-xe-dap/overview)
- Relevant research papers and articles on bicycle rental prediction
