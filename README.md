
# Credit Card Fraud Detection

A Streamlit-based web application that uses machine learning models to detect fraudulent credit card transactions.

## Project Overview

This application leverages two different machine learning models:
- Logistic Regression
- Random Forest

The models analyze transaction details to classify them as either legitimate or fraudulent.

## Features

- User-friendly web interface built with Streamlit
- Input fields for 29 transaction features (V1-V28 plus Amount)
- Real-time prediction using two different ML models
- Clear visual indicators for fraudulent transactions
- Standardized data processing for accurate predictions

## Outputs

The application's functionality is demonstrated through the following screenshots:

### Screenshot 1: Application Interface
![Application Interface](./outputs/1.png)
*The main interface showing the input fields for transaction details and prediction results.*

### Screenshot 2: Fraud Detection Example
![Fraud Detection Example](./outputs/2.png)
*Example of the application detecting a potentially fraudulent transaction.*

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies
```bash
pip install streamlit numpy scikit-learn joblib
```

3. Ensure you have the model files in the root directory:
   - `logistic_regression_model.pkl`
   - `random_forest_model.pkl`

## Usage

1. Start the Streamlit app
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter transaction details in the input fields

4. Click the "🔍 Predict Fraud" button to get the prediction results

## Technical Details

### Models
- **Logistic Regression**: A supervised learning classification algorithm used for predicting binary outcomes
- **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees during training

### Features
The application uses the standard credit card fraud detection dataset features:
- V1-V28: Principal component features (anonymized for privacy)
- Amount: Transaction amount

### Data Preprocessing
- Standardization of the 'Amount' feature using mean and standard deviation values from the training dataset

## Dataset Information

The models were trained on the Credit Card Fraud Detection dataset, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly unbalanced, with frauds accounting for 0.172% of all transactions.

Original dataset features:
- Time: Seconds elapsed between each transaction and the first transaction
- V1-V28: Principal components obtained with PCA
- Amount: Transaction amount
- Class: 1 for fraudulent transactions, 0 for legitimate ones

## Requirements

- Python 3.6+
- Streamlit
- NumPy
- scikit-learn
- joblib

## Future Improvements

- Add more machine learning models for comparison
- Implement confidence scores for predictions
- Create visualization of feature importance
- Add ability to upload CSV files for batch prediction
- Implement user authentication for secure access
- Add historical prediction tracking

## License

[MIT License](LICENSE)
